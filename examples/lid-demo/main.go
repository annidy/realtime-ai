package main

import (
	"bytes"
	"context"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"

	"github.com/realtime-ai/realtime-ai/pkg/connection"
	"github.com/realtime-ai/realtime-ai/pkg/elements"
	"github.com/realtime-ai/realtime-ai/pkg/pipeline"
	"github.com/realtime-ai/realtime-ai/pkg/server"
)

var (
	logger                  = log.New(os.Stderr, "[lid-demo]: ", log.LstdFlags|log.Lshortfile)
	_      pipeline.Element = (*LidElement)(nil)
)

type connectionEventHandler struct {
	connection.ConnectionEventHandler

	conn     connection.Connection
	pipeline *pipeline.Pipeline
}

func (c *connectionEventHandler) OnConnectionStateChange(state connection.ConnectionState) {
	log.Printf("Connection state changed: %v", state)

	if state == connection.ConnectionStateConnected {
		log.Println("WebRTC connection established")
	} else if state == connection.ConnectionStateFailed || state == connection.ConnectionStateClosed {
		log.Println("WebRTC connection ended")
		if c.pipeline != nil {
			c.pipeline.Stop()
		}
	}
}

func (c *connectionEventHandler) OnMessage(msg *pipeline.PipelineMessage) {
	// Push incoming audio to the pipeline
	c.pipeline.Push(msg)
}

func (c *connectionEventHandler) OnError(err error) {
	log.Printf("Connection error: %v", err)
}

func main() {
	// Create WebRTC server configuration
	cfg := &server.BasicWebRTCConfig{}
	cfg.RTCUDPPort = 9000
	cfg.ICELite = false

	// Create WebRTC server
	rtcServer := server.NewBasicWebRTCServer(cfg)

	// Set up connection handlers
	rtcServer.OnConnectionCreated(func(ctx context.Context, conn connection.Connection, r *http.Request) {
		log.Printf("New connection created: %s", conn.PeerID())

		// Create event handler
		eventHandler := &connectionEventHandler{
			conn: conn,
		}
		conn.RegisterEventHandler(eventHandler)

		// Create and configure pipeline
		p, _ := createPipeline(ctx, r)
		eventHandler.pipeline = p

		// Subscribe to pipeline events for logging
		// subscribeToEvents(conn, p)

		// Start pipeline
		if err := p.Start(ctx); err != nil {
			log.Printf("Failed to start pipeline: %v", err)
			return
		}

		// Start output handler
		go handlePipelineOutput(conn, p)

		log.Println("Pipeline started successfully")
	})

	// Start WebRTC server
	if err := rtcServer.Start(); err != nil {
		log.Fatalf("Failed to start WebRTC server: %v", err)
	}

	// Set up HTTP handlers
	http.HandleFunc("/session", rtcServer.HandleNegotiate)
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "examples/lid-demo/index.html")
	})

	// Start HTTP server in a goroutine
	go func() {
		log.Println("Starting HTTP server on :8080")
		log.Println("Open http://localhost:8080 in your browser")
		if err := http.ListenAndServe(":8080", nil); err != nil {
			log.Fatalf("Failed to start HTTP server: %v", err)
		}
	}()

	// Wait for interrupt signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	<-sigChan
	log.Println("\nShutting down...")
}

// createPipeline sets up the audio processing pipeline with VAD and Deepgram STT
func createPipeline(ctx context.Context, r *http.Request) (*pipeline.Pipeline, error) {
	p := pipeline.NewPipeline("lid-pipeline")

	// 1. Audio Resample Element (ensure 16kHz for VAD and Qwen)
	// AudioResampleElement(inputRate, outputRate, inputChannels, outputChannels)
	resampleElement := elements.NewAudioResampleElement(48000, 16000, 1, 1)
	p.AddElement(resampleElement)
	log.Println("Added: AudioResampleElement (48kHz → 16kHz, mono)")

	// 2. VAD Element
	var vadElement pipeline.Element
	vadConfig := elements.SileroVADConfig{
		ModelPath:       "models/silero_vad.onnx",
		Threshold:       0.5,
		MinSilenceDurMs: 300,
		SpeechPadMs:     30,
		Mode:            elements.VADModeFilter, // Passthrough mode forwards all audio
	}

	// Try to create VAD element (will fail gracefully if not built with vad tag)
	vadElem, err := elements.NewSileroVADElement(vadConfig)
	if err != nil {
		log.Fatalf("VAD not available (build with -tags vad to enable): %v", err)
	}
	if err := vadElem.Init(context.Background()); err != nil {
		log.Printf("[Pipeline] Warning: Failed to init VAD element: %v", err)
	} else {
		vadElement = vadElem
		p.AddElement(vadElement)
		log.Println("Added: SileroVADElement (Passthrough mode, emits events)")
	}

	lidElement := NewLidElement()
	p.AddElement(lidElement)

	// Pipeline: resample -> VAD -> Lid
	p.Link(resampleElement, vadElement)
	p.Link(vadElement, lidElement)

	log.Println("Pipeline configured successfully")
	return p, nil
}

func subscribeToEvents(conn connection.Connection, p *pipeline.Pipeline) {
	bus := p.Bus()
	if bus == nil {
		return
	}

	// Subscribe to VAD events
	vadEventsChan := make(chan pipeline.Event, 10)
	bus.Subscribe(pipeline.EventVADSpeechStart, vadEventsChan)
	bus.Subscribe(pipeline.EventVADSpeechEnd, vadEventsChan)

	// Handle VAD events
	go func() {
		for event := range vadEventsChan {
			switch event.Type {
			case pipeline.EventVADSpeechStart:
				log.Println("[VAD] Speech detected - streaming audio...")
			case pipeline.EventVADSpeechEnd:
				log.Println("[VAD] Speech ended - committing for final result...")
			}
		}
	}()
}

func handlePipelineOutput(conn connection.Connection, p *pipeline.Pipeline) {
	for {
		msg := p.Pull()
		if msg == nil {
			// Pipeline closed
			break
		}
		if msg.Type == pipeline.MsgTypeData && msg.TextData != nil {
			if msg.TextData.TextType == "application/json" {
				conn.SendMessage(msg)
			}
		}
	}
}

// LidElement is a simple element that echoes audio back.
type LidElement struct {
	*pipeline.BaseElement

	isProcess    atomic.Bool
	vadEventsSub chan pipeline.Event
	mu           sync.Mutex
	pcmBuf       bytes.Buffer
	writer       *io.PipeWriter

	// Lifecycle management
	ctx    context.Context
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// NewLidElement creates a new echo element.
func NewLidElement() *LidElement {
	return &LidElement{
		BaseElement: pipeline.NewBaseElement("lid", 100),
	}
}

// Start starts the echo element.
func (e *LidElement) Start(ctx context.Context) error {
	e.ctx, e.cancel = context.WithCancel(ctx)

	e.vadEventsSub = make(chan pipeline.Event, 10)
	e.Bus().Subscribe(pipeline.EventVADSpeechStart, e.vadEventsSub)
	e.Bus().Subscribe(pipeline.EventVADSpeechEnd, e.vadEventsSub)
	logger.Printf("Subscribed to VAD events")

	e.wg.Add(1)
	go e.processAudio(e.ctx)
	e.wg.Add(1)
	go e.handleVADEvents(e.ctx)
	logger.Printf("element started successfully")
	return nil
}

// Stop stops the echo element.
func (e *LidElement) Stop() error {
	logger.Printf("stopping element")
	if e.cancel != nil {
		e.cancel()
	}
	// Wait for goroutines
	e.wg.Wait()
	logger.Printf("element stopped")
	return nil
}

func (e *LidElement) processAudio(ctx context.Context) {
	defer e.wg.Done()
	defer func() {
		e.mu.Lock()
		if e.writer != nil {
			e.writer.Close()
			e.writer = nil
		}
		e.mu.Unlock()
	}()

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

			// With VAD, buffer audio and send when speaking
			e.mu.Lock()
			if e.writer != nil {
				e.writer.Write(msg.AudioData.Data)
			}
			e.mu.Unlock()
		}
	}
}

func (e *LidElement) handleVADEvents(ctx context.Context) {
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
				logger.Printf("VAD speech started")
				e.mu.Lock()
				if e.writer == nil {
					e.writer = e.createClient()
				}
				if e.writer != nil {
					// Extract pre-roll audio from VAD payload
					if payload, ok := event.Payload.(pipeline.VADPayload); ok {
						e.writer.Write(payload.PreRollAudio)
					}
				}
				e.mu.Unlock()

			case pipeline.EventVADSpeechEnd:
				logger.Printf("VAD speech ended")
				e.mu.Lock()
				if e.writer != nil {
					e.writer.Close()
					e.writer = nil
				}
				e.mu.Unlock()
			}
		}
	}
}

func (e *LidElement) createClient() *io.PipeWriter {
	if e.isProcess.Swap(true) {
		logger.Printf("http not finish, can't create client")
		return nil
	}
	// create underlying pipe
	reader, writer := io.Pipe()

	// HTTP goroutine: use the pipe reader as request body
	go func() {
		defer reader.Close()

		req, err := http.NewRequest("POST", "http://localhost:8000/predict", reader)
		if err != nil {
			logger.Printf("创建HTTP请求失败: %v", err)
			return
		}

		req = req.WithContext(e.ctx)

		req.Header.Set("Content-Type", "application/octet-stream")
		req.Header.Set("Accept", "*/*")
		logger.Println("发送请求")

		client := &http.Client{}
		resp, err := client.Do(req)
		if err != nil {
			logger.Printf("发送请求失败: %v", err)
			return
		}
		defer resp.Body.Close()

		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode != http.StatusOK {
			logger.Printf("服务器错误 %d: %s", resp.StatusCode, string(body))
			return
		}
		logger.Printf("响应: %s", string(body))
		e.OutChan <- &pipeline.PipelineMessage{
			Type: pipeline.MsgTypeData,
			TextData: &pipeline.TextData{
				Data:     body,
				TextType: "application/json",
			},
		}
		e.isProcess.Store(false)
	}()

	return writer
}
