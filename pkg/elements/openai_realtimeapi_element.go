package elements

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	openairt "github.com/WqyJh/go-openai-realtime/v2"
	"github.com/realtime-ai/realtime-ai/pkg/audio"
	"github.com/realtime-ai/realtime-ai/pkg/pipeline"
	"github.com/sashabaranov/go-openai"
)

// Make sure OpenAIRealtimeAPIElement implements pipeline.Element
var _ pipeline.Element = (*OpenAIRealtimeAPIElement)(nil)

type OpenAIRealtimeAPIConfig struct {
	Model string
}

type OpenAIRealtimeAPIElement struct {
	*pipeline.BaseElement

	model string

	conn      *openairt.Conn
	sessionID string
	dumper    *audio.Dumper

	cancel context.CancelFunc
	wg     sync.WaitGroup
}

func NewOpenAIRealtimeAPIElement(config OpenAIRealtimeAPIConfig) *OpenAIRealtimeAPIElement {
	var dumper *audio.Dumper
	var err error

	if config.Model == "" {
		config.Model = "gpt-realtime"
	}

	// 如果环境变量设置了，就创建dumper
	if os.Getenv("DUMP_OPENAI_AUDIO") == "true" {
		dumper, err = audio.NewDumper("openai_response", 24000, 1)
		if err != nil {
			log.Printf("create audio dumper error: %v", err)
		} else {
			log.Printf("Created audio dumper for OpenAI response")
		}
	}

	return &OpenAIRealtimeAPIElement{
		BaseElement: pipeline.NewBaseElement("openai-realtime-element", 100),
		dumper:      dumper,
		model:       config.Model,
	}
}

func (e *OpenAIRealtimeAPIElement) Start(ctx context.Context) error {

	apiKey := os.Getenv("OPENAI_API_KEY")
	// baseURL := os.Getenv("OPENAI_BASE_URL")

	client := openairt.NewClient(apiKey)
	// if baseURL != "" {
	// 	client.BaseURL = baseURL
	// }
	conn, err := client.Connect(context.Background(), openairt.WithModel(e.model))
	if err != nil {
		log.Printf("Failed to connect to OpenAI: %v", err)
		return err
	}
	e.conn = conn
	log.Println("openai connect success")

	responseDeltaHandler := func(ctx context.Context, event openairt.ServerEvent) {
		switch event.ServerEventType() {
		case openairt.ServerEventTypeResponseOutputAudioTranscriptDelta:
			// ignore
		case openairt.ServerEventTypeResponseOutputAudioTranscriptDone:
			fmt.Printf("[response] %s\n", event.(openairt.ResponseOutputAudioTranscriptDoneEvent).Transcript)
		}
	}

	// Full response
	responseHandler := func(ctx context.Context, event openairt.ServerEvent) {
		switch event.ServerEventType() {
		case openairt.ServerEventTypeConversationItemInputAudioTranscriptionCompleted:
			question := event.(openairt.ConversationItemInputAudioTranscriptionCompletedEvent).Transcript
			fmt.Printf("\n[question] %s\n", question)
		case openairt.ServerEventTypeResponseDone:
			fmt.Print("\n> ")
		}
	}

	// Log handler
	logHandler := func(ctx context.Context, event openairt.ServerEvent) {

		switch event.ServerEventType() {
		case openairt.ServerEventTypeError,
			openairt.ServerEventTypeSessionUpdated:
			data, err := json.Marshal(event)
			if err != nil {
				log.Fatal(err)
			}
			fmt.Printf("[%s] %s\n", event.ServerEventType(), string(data))
		}
	}

	audiobuffer := make([]byte, 0)

	audioResponseHandler := func(ctx context.Context, event openairt.ServerEvent) {
		switch event.ServerEventType() {
		case openairt.ServerEventTypeResponseOutputAudioDelta:
			msg := event.(openairt.ResponseOutputAudioDeltaEvent)
			// log.Printf("audioResponseHandler: %v", delta)
			data, err := base64.StdEncoding.DecodeString(msg.Delta)
			if err != nil {
				log.Fatal(err)
			}
			audiobuffer = append(audiobuffer, data...)

			// dump 音频数据
			if e.dumper != nil {
				if err := e.dumper.Write(data); err != nil {
					log.Printf("Failed to dump OpenAI audio: %v", err)
				}
			}

		case openairt.ServerEventTypeResponseOutputAudioDone:

			data := audiobuffer[:]
			audiobuffer = make([]byte, 0)

			e.BaseElement.OutChan <- &pipeline.PipelineMessage{
				Type:      pipeline.MsgTypeAudio,
				SessionID: e.sessionID,
				Timestamp: time.Now(),
				AudioData: &pipeline.AudioData{
					Data:       data,
					MediaType:  pipeline.AudioMediaTypeRaw,
					SampleRate: 24000, // AI 返回的采样率
					Channels:   1,     // AI 返回的通道数
					Timestamp:  time.Now(),
				},
			}

		case openairt.ServerEventTypeInputAudioBufferSpeechStarted:
			// Interrupt the current response
			msg := event.(openairt.InputAudioBufferSpeechStartedEvent)
			log.Println("AI session interrupted")
			e.Bus().Publish(pipeline.Event{
				Type:      pipeline.EventInterrupted,
				Timestamp: time.Now(),
				Payload:   msg,
			})
		case openairt.ServerEventTypeInputAudioBufferSpeechStopped:
			log.Println("Input audio buffer speech stopped")
		}
	}

	connHandler := openairt.NewConnHandler(ctx, conn, logHandler, responseHandler, responseDeltaHandler, audioResponseHandler)
	connHandler.Start()

	conn.SendMessage(ctx, openairt.SessionUpdateEvent{
		Session: openairt.SessionUnion{
			Realtime: &openairt.RealtimeSession{
				Model:            e.model,
				OutputModalities: []openairt.Modality{openairt.ModalityText, openairt.ModalityAudio},
				Audio: &openairt.RealtimeSessionAudio{
					Input: &openairt.SessionAudioInput{
						Format: &openairt.AudioFormatUnion{
							PCM: &openairt.AudioFormatPCM{
								Rate: 24000,
							},
						},
						Transcription: &openairt.AudioTranscription{
							Model: openai.Whisper1,
						},
						TurnDetection: &openairt.TurnDetectionUnion{
							ServerVad: &openairt.ServerVad{
								Threshold:         0.7,
								SilenceDurationMs: 800,
							},
						},
					},
					Output: &openairt.SessionAudioOutput{
						Format: &openairt.AudioFormatUnion{
							PCM: &openairt.AudioFormatPCM{
								Rate: 24000,
							},
						},
						Voice: openairt.VoiceShimmer,
					},
				},
				MaxOutputTokens: 4000,
			},
		},
	})
	log.Println("send update event")

	e.wg.Add(1)
	go func() {
		defer e.wg.Done()

		for {
			select {
			case <-ctx.Done():
				return
			case msg := <-e.BaseElement.InChan:

				if msg.Type == pipeline.MsgTypeAudio {

					if msg.AudioData.MediaType != pipeline.AudioMediaTypeRaw {
						continue
					}

					log.Printf("pipeline.MsgTypeAudio received, data length: %d", len(msg.AudioData.Data))

					// 将 PCM data 发送给 AI
					if len(msg.AudioData.Data) == 0 {
						continue
					}

					// 保存会话ID
					e.sessionID = msg.SessionID

					// 将 PCM data 发送给 AI
					if e.conn != nil {
						base64Audio := base64.StdEncoding.EncodeToString(msg.AudioData.Data)
						conn.SendMessage(ctx, openairt.InputAudioBufferAppendEvent{
							Audio: base64Audio,
						})
					}
				} else if msg.Type == pipeline.MsgTypeData {
					clientEvent, err := UnmarshalClientEvent(msg.TextData.Data)
					if err != nil {
						log.Println("AI session send error:", err)
						continue
					}

					if err := e.conn.SendMessage(ctx, clientEvent); err != nil {
						log.Println("AI session send error:", err)
						continue
					}
				} else {
					// 投递给下一环节
					e.BaseElement.OutChan <- msg
				}

			}
		}

	}()

	return nil
}

func (e *OpenAIRealtimeAPIElement) Stop() error {
	if e.cancel != nil {
		e.cancel()
		e.wg.Wait()
		e.cancel = nil
	}

	// 关闭 dumper
	if e.dumper != nil {
		e.dumper.Close()
		e.dumper = nil
	}

	return nil
}

// UnmarshalClientEvent unmarshals the client event from the given JSON data.
func UnmarshalClientEvent(data []byte) (openairt.ClientEvent, error) {
	var eventType struct {
		Type openairt.ClientEventType `json:"type"`
	}
	err := json.Unmarshal(data, &eventType)
	if err != nil {
		return nil, err
	}

	switch eventType.Type {
	case openairt.ClientEventTypeSessionUpdate:
		return unmarshalClientEvent[openairt.SessionUpdateEvent](data)
	case openairt.ClientEventTypeInputAudioBufferAppend:
		return unmarshalClientEvent[openairt.InputAudioBufferAppendEvent](data)
	case openairt.ClientEventTypeInputAudioBufferCommit:
		return unmarshalClientEvent[openairt.InputAudioBufferCommitEvent](data)
	case openairt.ClientEventTypeInputAudioBufferClear:
		return unmarshalClientEvent[openairt.InputAudioBufferClearEvent](data)
	case openairt.ClientEventTypeConversationItemCreate:
		return unmarshalClientEvent[openairt.ConversationItemCreateEvent](data)
	case openairt.ClientEventTypeConversationItemTruncate:
		return unmarshalClientEvent[openairt.ConversationItemTruncateEvent](data)
	case openairt.ClientEventTypeConversationItemDelete:
		return unmarshalClientEvent[openairt.ConversationItemDeleteEvent](data)
	case openairt.ClientEventTypeResponseCreate:
		return unmarshalClientEvent[openairt.ResponseCreateEvent](data)
	case openairt.ClientEventTypeResponseCancel:
		return unmarshalClientEvent[openairt.ResponseCancelEvent](data)
	}

	return nil, fmt.Errorf("unknown client event type: %s", eventType.Type)
}

// unmarshalClientEvent unmarshals the client event from the given JSON data.
func unmarshalClientEvent[T openairt.ClientEvent](data []byte) (T, error) {
	var t T
	err := json.Unmarshal(data, &t)
	if err != nil {
		return t, err
	}
	return t, nil
}
