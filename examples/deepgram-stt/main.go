package main

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/joho/godotenv"
	"github.com/realtime-ai/realtime-ai/pkg/connection"
	"github.com/realtime-ai/realtime-ai/pkg/elements"
	"github.com/realtime-ai/realtime-ai/pkg/pipeline"
	"github.com/realtime-ai/realtime-ai/pkg/server"
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
	// Load environment variables
	godotenv.Load()

	log.Println("=== Deegram STT Example with VAD Integration ===")
	log.Println("This example demonstrates:")
	log.Println("  - True streaming Speech-to-Text using Alibaba Cloud DashScope Qwen ASR")
	log.Println("  - Real-time partial and final transcription results")
	log.Println("  - Voice Activity Detection (VAD) using Silero")
	log.Println("  - Real-time audio processing pipeline")
	log.Println()
	log.Println("Deegram ASR uses WebSocket for true streaming,")
	log.Println()

	// Check for required API key
	if os.Getenv("DEEPGRAM_API_KEY") == "" {
		log.Fatal("DEEPGRAM_API_KEY environment variable is required")
	}

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
		subscribeToEvents(conn, p)

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
		http.ServeFile(w, r, "examples/deepgram-stt/index.html")
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
	p := pipeline.NewPipeline("deepgram-stt-pipeline")

	// Read model and language from request query params (sent by the client)
	model := r.URL.Query().Get("model")
	if model == "" {
		model = "nova-3"
	}
	lang := r.URL.Query().Get("lang")
	if lang == "" {
		lang = "en"
	}

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
		Mode:            elements.VADModePassthrough, // Passthrough mode forwards all audio
	}

	// Try to create VAD element (will fail gracefully if not built with vad tag)
	vadElem, err := elements.NewSileroVADElement(vadConfig)
	if err != nil {
		log.Printf("VAD not available (build with -tags vad to enable): %v", err)
		log.Println("Continuing without VAD optimization...")
	} else {
		if err := vadElem.Init(context.Background()); err != nil {
			log.Printf("[Pipeline] Warning: Failed to init VAD element: %v", err)
		} else {
			vadElement = vadElem
			p.AddElement(vadElement)
			log.Println("Added: SileroVADElement (Passthrough mode, emits events)")
		}
	}

	// 3. Deegram STT Element
	deepgramConfig := elements.DeepgramSTTConfig{
		APIKey:               os.Getenv("DEEPGROM_API_KEY"),
		Language:             lang,
		Model:                model,
		EnablePartialResults: true,  // Enable real-time partial results
		VADEnabled:           false, // Enable VAD integration if VAD is available
		SampleRate:           16000,
		Channels:             1,
		BitsPerSample:        16,
	}

	deepgramElement, err := elements.NewDeepgramSTTElement(deepgramConfig)
	if err != nil {
		log.Fatalf("Failed to create Deegram STT element: %v", err)
	}
	p.AddElement(deepgramElement)
	log.Printf("Added: DeepgramSTTElement (Language: %s, VAD: %v, Partial: %v)",
		deepgramConfig.Language, deepgramConfig.VADEnabled, deepgramConfig.EnablePartialResults)

	// Link elements together
	if vadElement != nil {
		// Pipeline: resample -> VAD -> Deegram STT
		p.Link(resampleElement, vadElement)
		p.Link(vadElement, deepgramElement)
	} else {
		// Pipeline: resample -> Deegram STT
		p.Link(resampleElement, deepgramElement)
	}

	log.Println("Pipeline configured successfully")
	return p, nil
}

// subscribeToEvents subscribes to pipeline events for monitoring
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
				jsonmsg := &pipeline.PipelineMessage{
					Type: pipeline.MsgTypeData,
					TextData: &pipeline.TextData{
						Data:      []byte("{\"speech\": true}"),
						Timestamp: time.Now(),
					},
				}
				conn.SendMessage(jsonmsg)
				log.Println("[VAD] Speech detected - streaming audio...")
			case pipeline.EventVADSpeechEnd:
				jsonmsg := &pipeline.PipelineMessage{
					Type: pipeline.MsgTypeData,
					TextData: &pipeline.TextData{
						Data:      []byte("{\"speech\": false}"),
						Timestamp: time.Now(),
					},
				}
				conn.SendMessage(jsonmsg)
				log.Println("[VAD] Speech ended - committing for final result...")
			}
		}
	}()
}

// handlePipelineOutput processes pipeline output and sends it back to the connection
func handlePipelineOutput(conn connection.Connection, p *pipeline.Pipeline) {
	for {
		msg := p.Pull()
		if msg == nil {
			// Pipeline closed
			break
		}

		// Log text data (transcriptions)
		if msg.Type == pipeline.MsgTypeData && msg.TextData != nil {
			text := string(msg.TextData.Data)
			if text != "" {
				textType := msg.TextData.TextType
				if textType == "text/final" {
					log.Printf("[Output] Final transcription: %s", text)
				} else {
					log.Printf("[Output] Partial transcription: %s", text)
				}
			}

			type TTSMessage struct {
				Type       string `json:"type"`
				Transcript string `json:"transcript"`
			}

			// Send message back to client (if needed)
			// For STT-only applications, you might send the text data back
			// Marshal to JSON
			jsonData, err := json.Marshal(TTSMessage{
				Type:       string(msg.TextData.TextType),
				Transcript: text,
			})
			if err != nil {
				log.Printf("Failed to marshal event: %v", err)
				return
			}

			// Create a text message with event data
			jsonmsg := &pipeline.PipelineMessage{
				Type: pipeline.MsgTypeData,
				TextData: &pipeline.TextData{
					Data: jsonData,
				},
			}
			conn.SendMessage(jsonmsg)
		}
	}

	log.Println("Output handler stopped")
}
