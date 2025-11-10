import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional
import wave

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.agents.llm import ChatContext, ChatMessage
from livekit.plugins import azure, silero
from openai import AsyncOpenAI

logger = logging.getLogger("agent")

load_dotenv(".env.local")


@dataclass(frozen=True)
class TargetLanguageConfig:
    id: str
    label: str
    translator_name: str
    azure_voice: str


TARGET_LANGUAGES: Dict[str, TargetLanguageConfig] = {
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

SOURCE_LANGUAGE_LABEL = "English"
DEFAULT_TARGET_LANGUAGE_ID = os.getenv("DEFAULT_TARGET_LANGUAGE", "hi-IN")


def resolve_target_language(language_id: str) -> TargetLanguageConfig:
    return TARGET_LANGUAGES.get(language_id, TARGET_LANGUAGES["hi-IN"])


@dataclass
class TranslationSegment:
    segment_id: str
    source_text: str
    translated_text: str
    target_language: TargetLanguageConfig


class ConversationMemory:
    def __init__(self, capacity: int = 4) -> None:
        self._segments: Deque[str] = deque(maxlen=capacity)

    def add(self, text: str) -> None:
        cleaned = text.strip()
        if cleaned:
            self._segments.append(cleaned)

    def context(self) -> List[str]:
        return list(self._segments)


class LiteLLMTranslator:
    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        timeout: float = 20.0,
    ) -> None:
        if not api_base or not api_key:
            raise ValueError("LITELLM_API_BASE and LITELLM_API_KEY must be set")

        self._client = AsyncOpenAI(base_url=api_base.rstrip("/"), api_key=api_key, timeout=timeout)
        self._model = model

    async def translate(
        self,
        text: str,
        *,
        target_language: TargetLanguageConfig,
        context: List[str],
    ) -> str:
        content = text.strip()
        if not content:
            return ""

        system_prompt = (
            "You are a real-time speech translation engine. Translate English speech segments into "
            f"{target_language.translator_name}. Use the provided context to keep tone, gendered references, "
            "and idioms consistent. Respond with only the translated text."
        )

        messages: List[dict] = [{"role": "system", "content": system_prompt}]

        if context:
            context_block = "\n".join(context[-3:])
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": "Previous English context to preserve meaning:\n" + context_block,
                    },
                    {
                        "role": "assistant",
                        "content": f"Context noted. Ready to translate into {target_language.translator_name}.",
                    },
                ]
            )

        messages.append(
            {
                "role": "user",
                "content": f"Translate this English segment into {target_language.translator_name}:\n{content}",
            }
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.2,
                max_tokens=256,
            )
        except Exception:
            logger.exception("LiteLLM translation request failed")
            raise

        translated = (response.choices[0].message.content or "").strip()
        return translated

    async def aclose(self) -> None:
        await self._client.close()


class TranslationAgent(Agent):
    def __init__(
        self,
        *,
        translator: LiteLLMTranslator,
        result_queue: "asyncio.Queue[TranslationSegment]",
        default_target: TargetLanguageConfig,
    ) -> None:
        super().__init__(
            instructions=(
                "You deliver speech-to-speech translation. The translation is computed externally; "
                "emit the provided translation without modification."
            )
        )
        self._translator = translator
        self._results = result_queue
        self._target_language = default_target
        self._memory = ConversationMemory()
        self._pending_segment: Optional[TranslationSegment] = None
        self._active_segment_id: Optional[str] = None
        self._lock = asyncio.Lock()

    @property
    def target_language(self) -> TargetLanguageConfig:
        return self._target_language

    async def set_target_language(self, config: TargetLanguageConfig) -> None:
        async with self._lock:
            self._target_language = config

    @property
    def active_segment_id(self) -> Optional[str]:
        return self._active_segment_id

    def clear_active_segment(self) -> None:
        self._active_segment_id = None

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        text_content = (new_message.text_content or "").strip() if new_message else ""
        if not text_content:
            return

        async with self._lock:
            target_language = self._target_language
            context = self._memory.context()

        try:
            translated_text = await self._translator.translate(
                text_content,
                target_language=target_language,
                context=context,
            )
        except Exception:
            translated_text = ""

        if not translated_text:
            logger.warning("Translation returned empty output; falling back to source text for playback.")

        segment = TranslationSegment(
            segment_id=uuid.uuid4().hex,
            source_text=text_content,
            translated_text=translated_text or text_content,
            target_language=target_language,
        )

        async with self._lock:
            self._pending_segment = segment
            self._memory.add(text_content)

        await self._results.put(segment)

    async def llm_node(
        self,
        chat_ctx: ChatContext,
        tools,
        model_settings,
    ):
        async with self._lock:
            segment = self._pending_segment
            self._pending_segment = None
            if segment:
                self._active_segment_id = segment.segment_id

        if not segment:
            logger.warning("No translation segment available for playback.")
            return

        yield segment.translated_text


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    logger.info("=" * 72)
    logger.info("Starting real-time translation agent")
    logger.info("Room: %s", ctx.room.name)
    logger.info("=" * 72)

    azure_speech_key = os.getenv("AZURE_SPEECH_KEY")
    azure_speech_region = os.getenv("AZURE_SPEECH_REGION")
    litellm_api_base = os.getenv("LITELLM_API_BASE")
    litellm_api_key = os.getenv("LITELLM_API_KEY")
    litellm_model = os.getenv("LITELLM_MODEL", "gpt-4.1")

    if not azure_speech_key or not azure_speech_region:
        raise RuntimeError("Azure speech credentials are required.")

    if not litellm_api_base or not litellm_api_key:
        raise RuntimeError("LiteLLM translation endpoint configuration is required.")

    recordings_root = Path(os.getenv("STT_RECORDINGS_DIR", "recordings"))
    session_folder = f"{ctx.room.name}-{int(time.time())}"
    session_recording_dir = recordings_root / session_folder
    session_recording_dir.mkdir(parents=True, exist_ok=True)
    logger.info("STT audio recordings will be written to %s", session_recording_dir)

    stt_silence_timeout_env = os.getenv("AZURE_STT_SILENCE_MS")
    stt_max_segment_env = os.getenv("AZURE_STT_MAX_MS")
    stt_silence_timeout = int(stt_silence_timeout_env) if stt_silence_timeout_env else None
    stt_max_segment_ms = int(stt_max_segment_env) if stt_max_segment_env else None

    default_language = resolve_target_language(DEFAULT_TARGET_LANGUAGE_ID)
    translator = LiteLLMTranslator(
        api_base=litellm_api_base,
        api_key=litellm_api_key,
        model=litellm_model,
    )

    translation_queue: asyncio.Queue[TranslationSegment] = asyncio.Queue()
    agent = TranslationAgent(
        translator=translator,
        result_queue=translation_queue,
        default_target=default_language,
    )

    recording_tasks: Dict[str, asyncio.Task] = {}

    azure_stt = azure.STT(
        speech_key=azure_speech_key,
        speech_region=azure_speech_region,
        language="en-US",
        segmentation_silence_timeout_ms=stt_silence_timeout,
        segmentation_max_time_ms=stt_max_segment_ms,
    )

    azure_tts = azure.TTS(
        speech_key=azure_speech_key,
        speech_region=azure_speech_region,
        voice=default_language.azure_voice,
        language=default_language.id,
    )

    session = AgentSession(
        stt=azure_stt,
        tts=azure_tts,
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
        min_endpointing_delay=0.35,
        max_endpointing_delay=4.5,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev) -> None:
        """Log aggregated usage reported by the session."""
        if hasattr(ev, "metrics"):
            metrics.log_metrics(ev.metrics)
            usage_collector.collect(ev.metrics)

    async def log_usage() -> None:
        summary = usage_collector.get_summary()
        logger.info("Usage summary: %s", summary)

    ctx.add_shutdown_callback(log_usage)

    async def publish_data(payload: dict) -> None:
        try:
            await ctx.room.local_participant.publish_data(
                json.dumps(payload).encode("utf-8"),
                reliable=True,
            )
        except Exception:
            logger.exception("Failed to publish payload: %s", payload.get("type"))

    async def publish_transcription(
        *,
        event_type: str,
        text: str,
        language_label: str,
        language_code: str,
        segment_id: str,
    ) -> None:
        if not text:
            return

        payload = {
            "type": event_type,
            "segment_id": segment_id,
            "language": language_label,
            "language_code": language_code,
            "text": text,
            "timestamp": time.time(),
        }
        await publish_data(payload)

    async def publish_segment_status(segment_id: Optional[str], status: str) -> None:
        if not segment_id:
            return
        await publish_data(
            {
                "type": "target_segment_status",
                "segment_id": segment_id,
                "status": status,
                "timestamp": time.time(),
            }
        )

    async def relay_translations() -> None:
        try:
            while True:
                segment = await translation_queue.get()
                await publish_transcription(
                    event_type="source_transcription",
                    text=segment.source_text,
                    language_label=SOURCE_LANGUAGE_LABEL,
                    language_code="en-US",
                    segment_id=segment.segment_id,
                )
                await publish_transcription(
                    event_type="target_transcription",
                    text=segment.translated_text,
                    language_label=segment.target_language.label,
                    language_code=segment.target_language.id,
                    segment_id=segment.segment_id,
                )
                translation_queue.task_done()
        except asyncio.CancelledError:
            logger.info("Translation relay cancelled.")

    relay_task = asyncio.create_task(relay_translations())

    def _is_audio_track(track) -> bool:
        track_kind = getattr(track, "kind", None)
        return track_kind in (rtc.TrackKind.KIND_AUDIO, "audio", 1)

    def _schedule_task_wait(task: asyncio.Task) -> None:
        async def _wait() -> None:
            with contextlib.suppress(asyncio.CancelledError):
                await task

        asyncio.create_task(_wait())

    def _start_audio_recording(track, participant_identity: str) -> None:
        if not _is_audio_track(track):
            return

        track_id = getattr(track, "sid", None) or str(id(track))
        existing = recording_tasks.pop(track_id, None)
        if existing:
            existing.cancel()
            _schedule_task_wait(existing)
        participant_dir = session_recording_dir / (participant_identity or "unknown")
        participant_dir.mkdir(parents=True, exist_ok=True)
        file_path = participant_dir / f"{int(time.time())}-{track_id}.wav"

        async def _record() -> None:
            try:
                stream = rtc.AudioStream(track=track)
            except Exception:
                logger.exception("Unable to open audio stream for %s", participant_identity)
                return

            wav_file = None
            try:
                async for event in stream:
                    if event is None:
                        continue

                    frame = event.frame
                    if wav_file is None:
                        wav_file = wave.open(str(file_path), "wb")
                        wav_file.setnchannels(frame.num_channels)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(frame.sample_rate)

                    wav_file.writeframes(frame.data.tobytes())
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error recording audio for %s", participant_identity)
            finally:
                if wav_file is not None:
                    wav_file.close()
                    logger.info("🎙️ Saved STT audio for %s to %s", participant_identity, file_path)
                await stream.aclose()

        task = asyncio.create_task(_record())
        recording_tasks[track_id] = task
        logger.info("🎙️ Recording started for %s into %s", participant_identity, file_path)

    async def shutdown_translator() -> None:
        relay_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await relay_task
        tasks = list(recording_tasks.values())
        recording_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await translator.aclose()

    ctx.add_shutdown_callback(shutdown_translator)

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant) -> None:
        participant_identity = getattr(participant, "identity", "unknown")
        logger.info(
            "🎧 Track subscribed: kind=%s codec=%s from %s",
            getattr(track, "kind", "unknown"),
            getattr(publication, "mime_type", "unknown"),
            participant_identity,
        )
        _start_audio_recording(track, participant_identity)

    @ctx.room.on("track_unsubscribed")
    def on_track_unsubscribed(track, publication, participant) -> None:
        participant_identity = getattr(participant, "identity", "unknown")
        logger.info(
            "🛑 Track unsubscribed: kind=%s from %s",
            getattr(track, "kind", "unknown"),
            participant_identity,
        )
        track_id = getattr(track, "sid", None) or str(id(track))
        task = recording_tasks.pop(track_id, None) if track_id else None
        if task:
            task.cancel()
            _schedule_task_wait(task)

    @session.on("user_speech_interim")
    def on_user_speech_interim(text: str) -> None:
        logger.info("🎤📝 [STT interim] %s", text)

    @session.on("agent_speech_interim")
    def on_agent_speech_interim(text: str) -> None:
        logger.info("🇮🇳📝 [Translation interim] %s", text)

    @session.on("tts_started")
    def on_tts_started() -> None:
        asyncio.create_task(publish_segment_status(agent.active_segment_id, "speaking"))

    @session.on("tts_finished")
    def on_tts_finished() -> None:
        asyncio.create_task(publish_segment_status(agent.active_segment_id, "completed"))
        agent.clear_active_segment()

    @session.on("error")
    def on_session_error(error: Exception) -> None:
        logger.exception("Session error", exc_info=error)

    async def apply_target_language(language_id: str) -> None:
        new_target = resolve_target_language(language_id)
        await agent.set_target_language(new_target)
        azure_tts.update_options(voice=new_target.azure_voice, language=new_target.id)
        await publish_data(
            {
                "type": "target_language_changed",
                "language_code": new_target.id,
                "language": new_target.label,
                "timestamp": time.time(),
            }
        )
        logger.info("Target language switched to %s", new_target.label)

    @ctx.room.on("data_received")
    def on_data_received(data: rtc.DataPacket) -> None:
        try:
            payload = json.loads(data.data.decode("utf-8"))
        except json.JSONDecodeError:
            sender = getattr(data.participant, "identity", "unknown")
            logger.warning("Received malformed data message from %s", sender)
            return

        message_type = payload.get("type")
        if message_type == "set_target_language":
            language_id = payload.get("language_id") or payload.get("languageCode")
            if language_id:
                asyncio.create_task(apply_target_language(language_id))
        elif message_type == "toggle_microphone":
            logger.debug("Microphone toggle command received: %s", payload)

    logger.info("Connecting to LiveKit room...")
    await ctx.connect()
    logger.info("LiveKit connection established. Starting session...")

    await session.start(agent=agent, room=ctx.room)
    logger.info("Agent session started.")

    await publish_data(
        {
            "type": "target_language_changed",
            "language_code": agent.target_language.id,
            "language": agent.target_language.label,
            "timestamp": time.time(),
        }
    )

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
