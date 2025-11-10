#!/usr/bin/env python3
"""Azure STT helper.

Run in two modes:

    python tests/test_azure_stt.py                       # connectivity only
    python tests/test_azure_stt.py --mode mic [--language hi-IN]

The "mic" mode streams the default microphone to Azure and prints interim/final
transcripts so you can see exactly what the backend would receive.
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Optional

import requests
from dotenv import load_dotenv

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:  # pragma: no cover - informative message for missing SDK
    speechsdk = None


load_dotenv(".env.local")


def _masked(value: Optional[str]) -> str:
    if not value:
        return "NOT SET"
    return f"{value[:8]}…"


def _credentials() -> tuple[str, str]:
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")
    if not speech_key or not speech_region:
        print("❌ AZURE_SPEECH_KEY/AZURE_SPEECH_REGION missing in .env.local")
        sys.exit(1)
    return speech_key, speech_region


def connectivity_check() -> bool:
    speech_key, speech_region = _credentials()

    print("=" * 80)
    print("Testing Azure Speech-to-Text (STT) Connectivity")
    print("=" * 80)
    print(f"Speech Key: {_masked(speech_key)}")
    print(f"Speech Region: {speech_region}")
    print()

    token_url = f"https://{speech_region}.api.cognitive.microsoft.com/sts/v1.0/issuetoken"
    headers = {"Ocp-Apim-Subscription-Key": speech_key, "Content-Length": "0"}

    try:
        response = requests.post(token_url, headers=headers, timeout=10)
        print(f"🔗 POST {token_url}")
        print(f"➡️  Status: {response.status_code}")

        if response.ok and response.text:
            print("✅ Token acquired; Azure STT is reachable.")
            print(f"   Token preview: {response.text[:8]}… (length={len(response.text)})")
            print("=" * 80)
            return True

        print("❌ Connection failed.")
        print(f"   Response: {response.text}")
        print("=" * 80)
        return False

    except requests.RequestException as exc:  # pragma: no cover - network path
        print("❌ Network request failed.")
        print(f"   Details: {exc}")
        print("=" * 80)
        return False


def microphone_session(language: str) -> None:
    if speechsdk is None:
        print("❌ azure-cognitiveservices-speech is not installed. Run `pip install azure-cognitiveservices-speech`.")
        sys.exit(1)

    speech_key, speech_region = _credentials()

    # Try to activate PulseAudio source if available (for WSL2/RDP scenarios)
    available_sources = []
    try:
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        source_name = parts[1]
                        available_sources.append(source_name)
                        # Try to activate each source
                        subprocess.run(
                            ["pactl", "suspend-source", source_name, "false"],
                            check=False,
                            capture_output=True,
                            timeout=1,
                        )
                        subprocess.run(
                            ["pactl", "set-source-mute", source_name, "0"],
                            check=False,
                            capture_output=True,
                            timeout=1,
                        )
            if available_sources:
                print(f"✅ Found {len(available_sources)} PulseAudio source(s): {', '.join(available_sources)}")
            else:
                print("⚠️  No PulseAudio sources found")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # pactl not available or timeout, continue anyway
        pass
    except Exception:
        # Ignore other errors
        pass

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = language

    # Try to create audio config with error handling
    # First try default microphone, then try specific device names if available
    audio_config = None
    device_names_to_try = []
    
    # Add PulseAudio source names to try
    for source in available_sources:
        if source and not source.endswith('.monitor'):
            device_names_to_try.append(source)
    
    # Try default first
    try:
        audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        if audio_config is None:
            print("❌ Failed to create audio config - no default microphone found")
        else:
            print("✅ Audio config created successfully (default microphone)")
    except Exception as e:
        print(f"⚠️  Failed to initialize audio config with default: {e}")
        audio_config = None
    
    # If default failed, try specific device names
    if audio_config is None and device_names_to_try:
        for device_name in device_names_to_try:
            try:
                print(f"   Trying device: {device_name}")
                audio_config = speechsdk.audio.AudioConfig(device_name=device_name)
                if audio_config:
                    print(f"✅ Audio config created successfully (device: {device_name})")
                    break
            except Exception as e:
                print(f"   Failed with device {device_name}: {e}")
                continue
    
    if audio_config is None:
        print("❌ Failed to create audio config with any method")
        print("   Azure Speech SDK cannot access microphone input.")
        print("   This is likely a PulseAudio/ALSA configuration issue on WSL2.")
        sys.exit(1)

    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    # Track session state
    session_active = False
    session_stopped_reason = None
    warning_printed = False

    print("=" * 80)
    print(f"🎤 Microphone streaming to Azure ({language})")
    print("   Speak into your default microphone. Press Ctrl+C to stop.")
    print("=" * 80)

    def on_recognizing(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        if evt.result.text:
            print(f"[interim] {evt.result.text}")

    def on_recognized(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            print(f"[final]   {evt.result.text}")
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            print("[final]   (no speech detected)")

    def on_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
        nonlocal session_stopped_reason
        session_stopped_reason = f"{evt.reason}: {evt.error_details or 'No details'}"
        print("❌ Canceled:", evt.reason, evt.error_details or "")
        if hasattr(evt, "error_code"):
            print(f"   Error code: {evt.error_code}")

    def on_speech_start_detected(evt) -> None:  # noqa: ANN001
        print("🔊 Audio detected - speech started")

    def on_speech_end_detected(evt) -> None:  # noqa: ANN001
        print("🔇 Audio ended - speech stopped")

    def on_session_started(evt) -> None:  # noqa: ANN001
        nonlocal session_active
        session_active = True
        print(f"[session] started {time.strftime('%H:%M:%S')}")
        if hasattr(evt, "session_id"):
            print(f"   Session ID: {evt.session_id}")

    def on_session_stopped(evt) -> None:  # noqa: ANN001
        nonlocal session_active, session_stopped_reason
        session_active = False
        if session_stopped_reason is None:
            session_stopped_reason = "Session stopped unexpectedly (no audio input detected)"
        print(f"[session] stopped {time.strftime('%H:%M:%S')}")
        print(f"   Reason: {session_stopped_reason}")
        # Debug: print all available attributes
        print(f"   Event type: {type(evt)}")
        print(f"   Event attributes: {[attr for attr in dir(evt) if not attr.startswith('_')]}")
        if hasattr(evt, "session_id"):
            print(f"   Session ID: {evt.session_id}")

    recognizer.recognizing.connect(on_recognizing)
    recognizer.recognized.connect(on_recognized)
    recognizer.canceled.connect(on_canceled)
    recognizer.speech_start_detected.connect(on_speech_start_detected)
    recognizer.speech_end_detected.connect(on_speech_end_detected)
    recognizer.session_started.connect(on_session_started)
    recognizer.session_stopped.connect(on_session_stopped)

    try:
        print("Starting continuous recognition...")
        recognizer.start_continuous_recognition()
        print("✅ Recognition started, waiting for audio...")
        
        # Wait and check if we're still running
        start_time = time.time()
        last_status_time = 0
        while True:
            time.sleep(0.1)
            elapsed = time.time() - start_time
            
            # Check if session is still active
            if not session_active and elapsed > 3 and not warning_printed:
                warning_printed = True
                print("\n⚠️  Session stopped but recognition loop still running.")
                print("   This usually means no audio input was detected.")
                print("   Check if your microphone is accessible to Azure Speech SDK.")
                print("   Try speaking now - if audio comes in, session may restart.")
            
            # Print status every 5 seconds
            if int(elapsed) - last_status_time >= 5:
                status = "active" if session_active else "inactive"
                print(f"[status] Session {status}, loop running ({int(elapsed)}s elapsed)...")
                last_status_time = int(elapsed)
                
    except KeyboardInterrupt:
        print("\nStopping recognition...")
    finally:
        try:
            recognizer.stop_continuous_recognition()
        except Exception as e:
            print(f"Error stopping recognition: {e}")
        print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="Azure STT diagnostic script")
    parser.add_argument(
        "--mode",
        choices=("connectivity", "mic"),
        default="connectivity",
        help="Connectivity check only (default) or microphone streaming test.",
    )
    parser.add_argument(
        "--language",
        default="en-US",
        help="BCP-47 locale for microphone streaming (default: en-US).",
    )
    args = parser.parse_args()

    if connectivity_check() and args.mode == "mic":
        microphone_session(args.language)

if __name__ == "__main__":
    main()