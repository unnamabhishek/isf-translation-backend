# Translation Agent - Backend

Python agent that powers real-time English → {Hindi | Telugu} speech translation using LiveKit, Azure Speech Services, and a LiteLLM-hosted GPT-4.x/GPT-5 model.

## Features

- Azure Speech-to-Text with low-latency segmentation for English
- LiteLLM gateway (GPT-4.x/GPT-5) for contextual translation
- Azure TTS voices for Hindi (`hi-IN`) and Telugu (`te-IN`)
- Real-time data channel streaming with segment IDs for frontend playback/highlights
- Dynamic target language switching without restarting the agent

## Setup

### 1. Install Dependencies

Using uv (recommended):
```bash
uv sync
```

Using pip:
```bash
pip install -e .
```

### 2. Configure Environment Variables

Copy the example file:
```bash
cp env-example.txt .env.local
```

Edit `.env.local` with your credentials:

```bash
# LiveKit Configuration
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret

# Azure Speech Service
AZURE_SPEECH_KEY=your-azure-speech-key
AZURE_SPEECH_REGION=your-azure-region

# Optional STT segmentation tuning (milliseconds)
AZURE_STT_SILENCE_MS=350
AZURE_STT_MAX_MS=4500

# LiteLLM Translation Gateway
LITELLM_API_BASE=https://your-litellm-gateway.example.com/v1
LITELLM_API_KEY=your-litellm-api-key
LITELLM_MODEL=gpt-4.1

# Default target voice (hi-IN or te-IN)
DEFAULT_TARGET_LANGUAGE=hi-IN
```

### 3. Run the Agent

Development mode (with auto-reload):
```bash
python src/agent.py dev
```

Production mode:
```bash
python src/agent.py start
```

## How It Works

1. **Listens** for English speech from LiveKit room participants
2. **Transcribes** using Azure Speech-to-Text (English)
3. **Translates** English segments using the LiteLLM gateway (GPT-4.x/GPT-5)
4. **Synthesizes** speech in the selected target language via Azure TTS
5. **Streams** audio back to the LiveKit room
6. **Publishes** source + target text segments (with IDs) to the frontend data channel
7. **Signals** playback state so the UI can highlight the spoken translation

## Configuration

### Change Target Voice

Update `TARGET_LANGUAGES` in `src/agent.py` to point at the Azure voice you prefer. Default voices:

```python
TARGET_LANGUAGES = {
    "hi-IN": TargetLanguageConfig(
        id="hi-IN",
        label="हिन्दी (Hindi)",
        translator_name="Hindi",
        azure_voice="hi-IN-SwaraNeural",
    ),
    "te-IN": TargetLanguageConfig(
        id="te-IN",
        label="తెలుగు (Telugu)",
        translator_name="Telugu",
        azure_voice="te-IN-ShrutiNeural",
    ),
}
```

See all available voices at:
https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support?tabs=tts

### Change Translation Model

Set `LITELLM_MODEL` in `.env.local` to the deployed model you want LiteLLM to target (for example `gpt-4.1`, `gpt-4o`, or your GPT-5 deployment name).

## Logs

The agent logs all activity including:
- Room connections
- English transcriptions
- Target-language translations
- Usage metrics

View logs in the console where you run the agent.

## Troubleshooting

### Import errors
- Make sure all dependencies are installed: `uv sync` or `pip install -e .`

### Azure authentication errors
- Verify your `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` are correct
- Check Azure portal for key validity

### LiteLLM errors
- Confirm the LiteLLM gateway URL is reachable from the agent environment
- Verify `LITELLM_API_KEY` permissions and expiry
- Check LiteLLM server logs for upstream provider issues

### LiveKit connection issues
- Verify `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and `LIVEKIT_API_SECRET`
- Check if LiveKit server is accessible

## Development

Run tests:
```bash
pytest
```

Format code:
```bash
ruff format src/
```

Lint code:
```bash
ruff check src/
```

## Architecture

```
User Speech (English)
    ↓
Azure STT
    ↓
LiteLLM (GPT-4.x/GPT-5) Translation
    ↓
Azure TTS (Hindi/Telugu)
    ↓
Target Audio Output
    ↘
Data Channel (source + translated segments + playback events)
```

Data flow is managed through LiveKit's real-time streaming infrastructure.
