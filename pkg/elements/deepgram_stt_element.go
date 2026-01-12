package elements

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	"github.com/realtime-ai/realtime-ai/pkg/asr"
	"github.com/realtime-ai/realtime-ai/pkg/pipeline"
)

var (
	logger                  = log.New(os.Stderr, "[DeepgramSTTElement]: ", log.LstdFlags|log.Lshortfile)
	_      pipeline.Element = (*DeepgramSTTElement)(nil)
)

type DeepgramSTTElement struct {
	*pipeline.BaseElement

	// ASR provider
	provider *asr.DeepgramProvider

	// ASR configuration
	language             string
	model                string
	enablePartialResults bool

	// Audio configuration (ElevenLabs requires 16kHz)
	sampleRate    int
	channels      int
	bitsPerSample int

	// VAD integration
	vadEnabled    bool
	vadEventsSub  chan pipeline.Event
	isSpeaking    bool
	speakingMutex sync.Mutex

	// Audio buffering (for VAD mode)
	audioBuffer     []byte
	audioBufferLock sync.Mutex

	// Streaming recognizer
	recognizer     asr.StreamingRecognizer
	recognizerLock sync.Mutex

	// Lifecycle management
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

type DeepgramSTTConfig struct {
	// APIKey is the Deepgram API key (if empty, will use DEEPGRAM_API_KEY env var)
	APIKey string

	// Language code (e.g., "en", "zh", "auto" for auto-detection)
	// Leave empty for auto-detection
	Language string

	// Model to use (default: "nova-3")
	Model string

	// EnablePartialResults enables interim results during recognition
	EnablePartialResults bool

	// VADEnabled determines if element should listen to VAD events
	// When true, recognition is triggered by VAD speech start/end events
	// When false, audio is sent continuously to recognizer
	VADEnabled bool

	// SampleRate in Hz (must be 16000 for Deepgram)
	SampleRate int

	// Channels (must be 1 for Deepgram - mono only)
	Channels int

	// BitsPerSample (default: 16)
	BitsPerSample int
}

func NewDeepgramSTTElement(config DeepgramSTTConfig) (*DeepgramSTTElement, error) {
	// Get API key from config or environment
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("DEEPGRAM_API_KEY")
	}

	if apiKey == "" {
		return nil, fmt.Errorf("Deepgram API key is required (set APIKey or DEEPGRAM_API_KEY env var)")
	}

	// Create Deepgram provider
	provider := asr.NewDeepgramProvider(apiKey)

	// Set defaults and validate
	if config.SampleRate == 0 {
		config.SampleRate = 16000
	}
	if config.SampleRate != 16000 {
		return nil, fmt.Errorf("Deepgram requires 16kHz sample rate, got %d", config.SampleRate)
	}

	if config.Channels == 0 {
		config.Channels = 1
	}
	if config.Channels != 1 {
		return nil, fmt.Errorf("Deepgram only supports mono audio, got %d channels", config.Channels)
	}

	if config.BitsPerSample == 0 {
		config.BitsPerSample = 16
	}

	elem := &DeepgramSTTElement{
		BaseElement:          pipeline.NewBaseElement("deepgram-stt", 100),
		provider:             provider,
		language:             config.Language,
		model:                config.Model,
		enablePartialResults: config.EnablePartialResults,
		vadEnabled:           config.VADEnabled,
		sampleRate:           config.SampleRate,
		channels:             config.Channels,
		bitsPerSample:        config.BitsPerSample,
		audioBuffer:          make([]byte, 0, 16000*2*10), // 10 seconds buffer
	}

	return elem, nil
}

// Start starts the Deepgram STT element.
func (e *DeepgramSTTElement) Start(ctx context.Context) error {
	e.ctx, e.cancel = context.WithCancel(ctx)

	logger.Printf("Starting element (VAD: %v, Language: %s, Model: %s)",
		e.vadEnabled, e.language, e.model)

	// Subscribe to VAD events if VAD is enabled
	if e.vadEnabled && e.BaseElement.Bus() != nil {
		e.vadEventsSub = make(chan pipeline.Event, 10)
		e.BaseElement.Bus().Subscribe(pipeline.EventVADSpeechStart, e.vadEventsSub)
		e.BaseElement.Bus().Subscribe(pipeline.EventVADSpeechEnd, e.vadEventsSub)

		logger.Printf("Subscribed to VAD events")
	}

	// Start streaming recognizer
	if err := e.startRecognizer(e.ctx); err != nil {
		e.cancel()
		return fmt.Errorf("failed to start recognizer: %w", err)
	}

	// Start audio processing goroutine
	e.wg.Add(1)
	go e.processAudio(e.ctx)

	// Start VAD event handler if enabled
	if e.vadEnabled {
		e.wg.Add(1)
		go e.handleVADEvents(e.ctx)
	}

	// Start result handler
	e.wg.Add(1)
	go e.handleResults(e.ctx)

	logger.Printf("element started successfully")
	return nil
}

// Stop stops the Deepgram STT element.
func (e *DeepgramSTTElement) Stop() error {
	logger.Printf("stopping element")

	if e.cancel != nil {
		e.cancel()
	}

	// Close recognizer first
	e.recognizerLock.Lock()
	if e.recognizer != nil {
		e.recognizer.Close()
		e.recognizer = nil
	}
	e.recognizerLock.Unlock()

	// Wait for goroutines
	e.wg.Wait()

	// Close provider
	if e.provider != nil {
		e.provider.Close()
	}

	// Unsubscribe from VAD events
	if e.vadEventsSub != nil {
		if e.BaseElement.Bus() != nil {
			e.BaseElement.Bus().Unsubscribe(pipeline.EventVADSpeechStart, e.vadEventsSub)
			e.BaseElement.Bus().Unsubscribe(pipeline.EventVADSpeechEnd, e.vadEventsSub)
		}
		close(e.vadEventsSub)
		e.vadEventsSub = nil
	}

	logger.Printf("element stopped")
	return nil
}

// startRecognizer creates and starts a streaming recognizer.
func (e *DeepgramSTTElement) startRecognizer(ctx context.Context) error {
	e.recognizerLock.Lock()
	defer e.recognizerLock.Unlock()

	audioConfig := asr.AudioConfig{
		SampleRate:    e.sampleRate,
		Channels:      e.channels,
		Encoding:      "pcm",
		BitsPerSample: e.bitsPerSample,
	}

	recognitionConfig := asr.RecognitionConfig{
		Language:             e.language,
		Model:                e.model,
		EnablePartialResults: e.enablePartialResults,
	}

	recognizer, err := e.provider.StreamingRecognize(ctx, audioConfig, recognitionConfig)
	if err != nil {
		return fmt.Errorf("failed to create streaming recognizer: %w", err)
	}

	e.recognizer = recognizer
	logger.Printf("Streaming recognizer started")
	return nil
}

// processAudio processes incoming audio messages.
func (e *DeepgramSTTElement) processAudio(ctx context.Context) {
	defer e.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return

		case msg, ok := <-e.BaseElement.InChan:
			if !ok {
				return
			}

			// Only process audio messages
			if msg.Type != pipeline.MsgTypeAudio || msg.AudioData == nil {
				continue
			}

			// Validate audio format
			if msg.AudioData.SampleRate != e.sampleRate {
				logger.Printf("Warning: Audio sample rate mismatch (expected %d, got %d)",
					e.sampleRate, msg.AudioData.SampleRate)
				continue
			}

			// If VAD is disabled, send audio directly to recognizer
			if !e.vadEnabled {
				e.sendAudioToRecognizer(ctx, msg.AudioData.Data)
			} else {
				// With VAD, buffer audio and send when speaking
				e.speakingMutex.Lock()
				isSpeaking := e.isSpeaking
				e.speakingMutex.Unlock()

				if isSpeaking {
					// Buffer audio for potential commit
					e.audioBufferLock.Lock()
					e.audioBuffer = append(e.audioBuffer, msg.AudioData.Data...)
					e.audioBufferLock.Unlock()

					// Send audio to recognizer
					e.sendAudioToRecognizer(ctx, msg.AudioData.Data)
				}
			}
		}
	}
}

// handleVADEvents processes VAD speech start/end events.
func (e *DeepgramSTTElement) handleVADEvents(ctx context.Context) {
	defer e.wg.Done()

	for {
		select {
		case <-ctx.Done():
			return

		case event, ok := <-e.vadEventsSub:
			if !ok {
				return
			}

			switch event.Type {
			case pipeline.EventVADSpeechStart:
				// Extract pre-roll audio from VAD payload
				if payload, ok := event.Payload.(pipeline.VADPayload); ok {
					// Send pre-roll audio first (before setting isSpeaking)
					if len(payload.PreRollAudio) > 0 {
						logger.Printf("VAD speech started with %d bytes pre-roll audio",
							len(payload.PreRollAudio))
						e.sendAudioToRecognizer(ctx, payload.PreRollAudio)
					} else {
						logger.Printf("VAD speech started (no pre-roll)")
					}
				} else {
					logger.Printf("VAD speech started (legacy payload)")
				}

				e.speakingMutex.Lock()
				e.isSpeaking = true
				e.speakingMutex.Unlock()

				// Clear buffer and start fresh
				e.audioBufferLock.Lock()
				e.audioBuffer = e.audioBuffer[:0]
				e.audioBufferLock.Unlock()

			case pipeline.EventVADSpeechEnd:
				logger.Printf("VAD speech ended")
				e.speakingMutex.Lock()
				e.isSpeaking = false
				e.speakingMutex.Unlock()

				// Commit to trigger final transcription
				e.commitRecognizer(ctx)
			}
		}
	}
}

// sendAudioToRecognizer sends audio data to the streaming recognizer.
func (e *DeepgramSTTElement) sendAudioToRecognizer(ctx context.Context, audioData []byte) {
	e.recognizerLock.Lock()
	recognizer := e.recognizer
	e.recognizerLock.Unlock()

	if recognizer == nil {
		logger.Printf("No recognizer available")
		return
	}

	if err := recognizer.SendAudio(ctx, audioData); err != nil {
		logger.Printf("Error sending audio to recognizer: %v", err)
	}
}

// commitRecognizer commits the audio buffer to trigger final transcription.
func (e *DeepgramSTTElement) commitRecognizer(ctx context.Context) {
	e.recognizerLock.Lock()
	recognizer := e.recognizer
	e.recognizerLock.Unlock()

	if recognizer == nil {
		return
	}
	// TODO: recognizer.Commit() not implemented
}

// handleResults processes recognition results from the streaming recognizer.
func (e *DeepgramSTTElement) handleResults(ctx context.Context) {
	defer e.wg.Done()

	e.recognizerLock.Lock()
	recognizer := e.recognizer
	e.recognizerLock.Unlock()

	if recognizer == nil {
		logger.Printf("No recognizer available for results")
		return
	}

	resultsChan := recognizer.Results()

	for {
		select {
		case <-ctx.Done():
			return

		case result, ok := <-resultsChan:
			if !ok {
				logger.Printf("Results channel closed")
				return
			}

			if result == nil {
				continue
			}

			// Skip empty results
			if result.Text == "" {
				continue
			}

			// Determine text type
			textType := pipeline.TextDataPartialType
			if result.IsFinal {
				textType = pipeline.TextDataFinalType
			}

			logger.Printf("Recognition result (%s): %s", textType, result.Text)

			// Create text data message
			textMsg := &pipeline.PipelineMessage{
				Type:      pipeline.MsgTypeData,
				Timestamp: time.Now(),
				TextData: &pipeline.TextData{
					Data:      []byte(result.Text),
					TextType:  textType,
					Timestamp: result.Timestamp,
				},
			}

			// Send to output channel
			select {
			case e.BaseElement.OutChan <- textMsg:
			case <-ctx.Done():
				return
			}

			// Publish event to bus
			if result.IsFinal {
				e.Bus().Publish(pipeline.Event{
					Type:      pipeline.EventPartialResult,
					Timestamp: result.Timestamp,
					Payload: pipeline.FinalResultPayload{
						Transcript: result.Text,
					},
				})
			} else {
				e.Bus().Publish(pipeline.Event{
					Type:      pipeline.EventFinalResult,
					Timestamp: result.Timestamp,
					Payload: pipeline.PartialResultPayload{
						Delta: result.Text,
					},
				})
			}
		}
	}
}
