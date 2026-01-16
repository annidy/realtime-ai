package asr

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	msginterfaces "github.com/deepgram/deepgram-go-sdk/v3/pkg/api/listen/v1/websocket/interfaces"
	"github.com/deepgram/deepgram-go-sdk/v3/pkg/client/interfaces"
	client "github.com/deepgram/deepgram-go-sdk/v3/pkg/client/listen"
	listenv1ws "github.com/deepgram/deepgram-go-sdk/v3/pkg/client/listen/v1/websocket"
)

var (
	logger     = log.New(os.Stderr, "[Deepgram]: ", log.LstdFlags|log.Lshortfile)
	clientInit sync.Once
)

type DeepgramProvider struct {
	apiKey string
	mu     sync.RWMutex
}

func NewDeepgramProvider(apiKey string) *DeepgramProvider {
	clientInit.Do(func() {
		client.Init(client.InitLib{
			LogLevel: client.LogLevelDebug,
		})
	})
	return &DeepgramProvider{
		apiKey: apiKey,
	}
}

func (p *DeepgramProvider) Name() string {
	return "deepgram"
}

func (p *DeepgramProvider) Recognize(ctx context.Context, audio io.Reader, audioConfig AudioConfig, config RecognitionConfig) (*RecognitionResult, error) {
	return nil, errors.New("Deepgram ASR not implemented yet")
}

func (p *DeepgramProvider) StreamingRecognize(ctx context.Context, audioConfig AudioConfig, config RecognitionConfig) (StreamingRecognizer, error) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	recognizer := &deepgramStreamingRecognizer{
		provider:    p,
		audioConfig: audioConfig,
		config:      config,
		resultsChan: make(chan *RecognitionResult, 10),
		sendChan:    make(chan []byte, 100),
		commitChan:  make(chan struct{}, 1),
		openChan:    make(chan *msginterfaces.OpenResponse),
		// msginterfaces.LiveMessageChan
		messageChan:       make(chan *msginterfaces.MessageResponse),
		metadataChan:      make(chan *msginterfaces.MetadataResponse),
		speechStartedChan: make(chan *msginterfaces.SpeechStartedResponse),
		utteranceEndChan:  make(chan *msginterfaces.UtteranceEndResponse),
		closeChan:         make(chan *msginterfaces.CloseResponse),
		errorChan:         make(chan *msginterfaces.ErrorResponse),
		unhandledChan:     make(chan *[]byte),
	}

	// Start connection
	if err := recognizer.connect(ctx); err != nil {
		return nil, err
	}
	logger.Println("Deepgram connected")
	return recognizer, nil
}

func (p *DeepgramProvider) SupportsStreaming() bool {
	return true
}

func (p *DeepgramProvider) SupportedLanguages() []string {
	return []string{
		"en", "zh", "es", "fr", "de", "it", "pt", "ru", "ja", "ko",
		"ar", "hi", "nl", "pl", "tr", "vi", "th", "id", "auto",
	}
}

func (p *DeepgramProvider) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	return nil
}

// Deepgram 核心实现
type deepgramStreamingRecognizer struct {
	provider    *DeepgramProvider
	audioConfig AudioConfig
	config      RecognitionConfig
	resultsChan chan *RecognitionResult
	sendChan    chan []byte
	commitChan  chan struct{}
	mu          sync.Mutex
	ctx         context.Context
	cancel      context.CancelFunc
	closed      atomic.Bool

	dgClient *listenv1ws.WSChannel

	// msginterfaces.LiveMessageChan
	openChan          chan *msginterfaces.OpenResponse
	messageChan       chan *msginterfaces.MessageResponse
	metadataChan      chan *msginterfaces.MetadataResponse
	speechStartedChan chan *msginterfaces.SpeechStartedResponse
	utteranceEndChan  chan *msginterfaces.UtteranceEndResponse
	closeChan         chan *msginterfaces.CloseResponse
	errorChan         chan *msginterfaces.ErrorResponse
	unhandledChan     chan *[]byte
}

// GetOpen returns the open channels
func (dch *deepgramStreamingRecognizer) GetOpen() []*chan *msginterfaces.OpenResponse {
	return []*chan *msginterfaces.OpenResponse{&dch.openChan}
}

// GetMessage returns the message channels
func (dch *deepgramStreamingRecognizer) GetMessage() []*chan *msginterfaces.MessageResponse {
	return []*chan *msginterfaces.MessageResponse{&dch.messageChan}
}

// GetMetadata returns the metadata channels
func (dch *deepgramStreamingRecognizer) GetMetadata() []*chan *msginterfaces.MetadataResponse {
	return []*chan *msginterfaces.MetadataResponse{&dch.metadataChan}
}

// GetSpeechStarted returns the speech started channels
func (dch *deepgramStreamingRecognizer) GetSpeechStarted() []*chan *msginterfaces.SpeechStartedResponse {
	return []*chan *msginterfaces.SpeechStartedResponse{&dch.speechStartedChan}
}

// GetUtteranceEnd returns the utterance end channels
func (dch *deepgramStreamingRecognizer) GetUtteranceEnd() []*chan *msginterfaces.UtteranceEndResponse {
	return []*chan *msginterfaces.UtteranceEndResponse{&dch.utteranceEndChan}
}

// GetClose returns the close channels
func (dch *deepgramStreamingRecognizer) GetClose() []*chan *msginterfaces.CloseResponse {
	return []*chan *msginterfaces.CloseResponse{&dch.closeChan}
}

// GetError returns the error channels
func (dch *deepgramStreamingRecognizer) GetError() []*chan *msginterfaces.ErrorResponse {
	return []*chan *msginterfaces.ErrorResponse{&dch.errorChan}
}

// GetUnhandled returns the unhandled event channels
func (dch *deepgramStreamingRecognizer) GetUnhandled() []*chan *[]byte {
	return []*chan *[]byte{&dch.unhandledChan}
}

// Open is the callback for when the connection opens
// golintci: funlen
func (dch *deepgramStreamingRecognizer) Run() error {
	wgReceivers := sync.WaitGroup{}

	// open channel
	wgReceivers.Add(1)
	go func() {
		defer wgReceivers.Done()

		for range dch.openChan {
			fmt.Printf("\n[OpenResponse]\n")
		}
	}()

	// message channel
	wgReceivers.Add(1)
	go func() {
		defer wgReceivers.Done()
		var tmpSentence strings.Builder
		for mr := range dch.messageChan {
			sentence := strings.TrimSpace(mr.Channel.Alternatives[0].Transcript)

			if len(mr.Channel.Alternatives) == 0 || sentence == "" {
				logger.Println("DEEPGRAM - no transcript")
				continue
			}

			if mr.IsFinal {
				logger.Printf("\n[MessageResponse] (Final) %s\n", sentence)
			} else {
				logger.Printf("\n[MessageResponse] (Interim) %s\n", sentence)
			}

			result := &RecognitionResult{
				Text:       tmpSentence.String() + sentence,
				IsFinal:    mr.IsFinal && mr.SpeechFinal,
				Confidence: float32(mr.Channel.Alternatives[0].Confidence),
				Timestamp:  time.Now(),
				Metadata:   map[string]interface{}{},
			}
			if mr.SpeechFinal {
				tmpSentence.Reset()
			} else if mr.IsFinal {
				tmpSentence.WriteString(sentence)
				logger.Print("\n\n ************* \n\n")
			}

			select {
			case dch.resultsChan <- result:
			case <-dch.ctx.Done():
			default:
				logger.Printf("Results channel full, dropping message")
			}
		}
	}()

	// metadata channel
	wgReceivers.Add(1)
	go func() {
		defer wgReceivers.Done()

		for mr := range dch.metadataChan {
			logger.Printf("\n\nMetadata.RequestID: %s\n", strings.TrimSpace(mr.RequestID))
			logger.Printf("Metadata.Channels: %d\n", mr.Channels)
			logger.Printf("Metadata.Created: %s\n\n", strings.TrimSpace(mr.Created))
		}
	}()

	// speech started channel
	wgReceivers.Add(1)
	go func() {
		defer wgReceivers.Done()

		for _ = range dch.speechStartedChan {
			logger.Printf("\n[SpeechStarted]\n")
		}
	}()

	// utterance end channel
	wgReceivers.Add(1)
	go func() {
		defer wgReceivers.Done()

		for range dch.utteranceEndChan {
			logger.Printf("\n[UtteranceEnd]\n")
		}
	}()

	// close channel
	wgReceivers.Add(1)
	go func() {
		defer wgReceivers.Done()

		for range dch.closeChan {
			logger.Printf("\n\n[CloseResponse]\n\n")
		}
	}()

	// error channel
	wgReceivers.Add(1)
	go func() {
		defer wgReceivers.Done()

		for er := range dch.errorChan {
			logger.Printf("\n[ErrorResponse]\n")
			logger.Printf("\nError.Type: %s\n", er.ErrCode)
			logger.Printf("Error.Message: %s\n", er.ErrMsg)
			logger.Printf("Error.Description: %s\n\n", er.Description)
			logger.Printf("Error.Variant: %s\n\n", er.Variant)
		}
	}()

	// unhandled event channel
	wgReceivers.Add(1)
	go func() {
		defer wgReceivers.Done()

		for byData := range dch.unhandledChan {
			logger.Printf("\n[UnhandledEvent]")
			logger.Printf("Dump:\n%s\n\n", string(*byData))
		}
	}()

	// wait for all receivers to finish
	wgReceivers.Wait()

	return nil
}

func (r *deepgramStreamingRecognizer) connect(ctx context.Context) error {
	r.ctx, r.cancel = context.WithCancel(ctx)

	logger.Println("DEEPGRAM - connecting")
	// Initialize Deepgram client
	// client options
	cOptions := &interfaces.ClientOptions{
		EnableKeepAlive: true,
	}

	// set the Transcription options
	tOptions := &interfaces.LiveTranscriptionOptions{
		Model:       r.config.Model,    // "nova-3",
		Keyterm:     []string{},        // Keyterm only support in nova-3
		Language:    r.config.Language, // "en",
		Punctuate:   true,
		Encoding:    "linear16",
		Channels:    1,
		SampleRate:  r.audioConfig.SampleRate, // 16000,
		SmartFormat: true,                     // additional formatting will be applied to transcripts to improve readability
		VadEvents:   false,                    // no use VAD events
		// To get UtteranceEnd, the following must be set:
		InterimResults: true,
		UtteranceEndMs: "1000",
		// End of UtteranceEnd settings
	}

	// implement your own callback
	var callback msginterfaces.LiveMessageChan
	callback = r
	go func() {
		r.Run()
	}()
	// create a Deepgram client
	dgClient, err := client.NewWSUsingChan(ctx, r.provider.apiKey, cOptions, tOptions, callback)
	if err != nil {
		logger.Println("ERROR creating LiveTranscription connection:", err)
		return err
	}
	// connect the websocket to Deepgram
	bConnected := dgClient.Connect()
	if !bConnected {
		return errors.New("Deepgram client connect failed")
	}
	r.dgClient = dgClient

	return nil
}

// Recognizer interface

// SendAudio sends audio data to the recognizer.
func (r *deepgramStreamingRecognizer) SendAudio(ctx context.Context, audioData []byte) error {
	if r.closed.Load() {
		return &Error{
			Code:    ErrCodeProviderError,
			Message: "recognizer is closed",
		}
	}
	_, err := r.dgClient.Write(audioData)
	return err
}

// Results returns a channel that receives recognition results.
func (r *deepgramStreamingRecognizer) Results() <-chan *RecognitionResult {
	return r.resultsChan
}

// Close stops recognition and releases resources.
func (r *deepgramStreamingRecognizer) Close() error {
	if r.closed.Swap(true) {
		return nil // Already closed
	}
	logger.Println("closing recognizer")

	if r.cancel != nil {
		r.cancel()
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.dgClient.Stop()
	close(r.resultsChan)

	logger.Println("recognizer closed")
	return nil
}
