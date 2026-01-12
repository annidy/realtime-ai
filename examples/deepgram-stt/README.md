# Deepgram STT Example

This example demonstrates real-time speech-to-text using Deepgram STT with optional VAD (Voice Activity Detection) integration.

## Features

- **True Streaming ASR**: Uses WebSocket for real-time audio streaming (similar to OpenAI Realtime API)
- **Partial Results**: Get interim transcription results as you speak
- **Final Results**: Get complete transcription when speech ends
- **VAD Integration**: Optional Silero VAD for optimized recognition
- **Multiple Languages**: Supports Chinese (zh), English (en), Japanese (ja), Korean (ko), Cantonese (yue)

## Prerequisites

1. **DashScope API Key**: Get your API key from [Alibaba Cloud DashScope](https://dashscope.console.aliyun.com/)

2. **Set environment variable**:
   ```bash
   export DEEPGRAM_API_KEY=your_api_key_here
   ```

## Running the Example

### WebRTC Server Mode (Browser)

```bash
go run examples/qwen-realtime-stt/main.go
```

### Audio File Mode (CLI)

This mode uses the provided `test.m4a` for testing:

```bash
go run examples/qwen-realtime-stt/test_audio/main.go
```

### With VAD (Requires ONNX Runtime)

```bash
# First, set up ONNX Runtime (see pkg/elements/VAD_README.md)
go run -tags vad examples/qwen-realtime-stt/main.go
```

## How It Works

### Pipeline Architecture

```
┌──────────────────┐
│  WebRTC Audio    │
│  (Browser)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│ AudioResampleElement │
│  → 16kHz, mono       │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  SileroVADElement    │  (Optional)
│  Mode: Passthrough   │
│  Events: ─────────┐  │
└────────┬──────────┘  │
         │             │
         ▼             │ EventVADSpeechStart
┌──────────────────────┼───────┐
│ DeepgramSTTElement           │
│  - WebSocket streaming       │
│  - Partial results: ────────►│── Partial text
│  - Final results: ─────────►│── Final text
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│  TextData Output             │
│  → Browser / Further         │
│    Processing                │
└──────────────────────────────┘
```

### Event Flow

1. **Without VAD**:
   - Audio is streamed continuously to Qwen Realtime
   - Partial results are emitted in real-time
   - Manual commit triggers final transcription

2. **With VAD**:
   - VAD detects speech start → Audio streaming begins
   - Partial results are emitted during speech
   - VAD detects speech end → Commits audio buffer
   - Final transcription is returned

