#!/usr/bin/env python3
"""
Test Azure Text-to-Speech (TTS)
Run this to verify Azure TTS is working with your credentials
"""

import os
from typing import Optional

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.local")


def _masked(value: Optional[str]) -> str:
    if not value:
        return "NOT SET"
    return f"{value[:8]}…"


def test_azure_tts():
    """Perform a lightweight Azure Text-to-Speech connectivity check."""
    
    print("=" * 80)
    print("Testing Azure Text-to-Speech (TTS)")
    print("=" * 80)
    
    # Get credentials
    speech_key = os.getenv("AZURE_SPEECH_KEY")
    speech_region = os.getenv("AZURE_SPEECH_REGION")
    voice_name = os.getenv("AZURE_TTS_VOICE", "hi-IN-SwaraNeural")

    print(f"Speech Key: {_masked(speech_key)}")
    print(f"Speech Region: {speech_region}")
    print(f"Voice: {voice_name}")
    print()
    
    if not speech_key or not speech_region:
        print("❌ ERROR: AZURE_SPEECH_KEY or AZURE_SPEECH_REGION not set in .env.local")
        return
    
    voices_url = f"https://{speech_region}.tts.speech.microsoft.com/cognitiveservices/voices/list"
    headers = {
        "Ocp-Apim-Subscription-Key": speech_key,
    }

    try:
        response = requests.get(voices_url, headers=headers, timeout=10)
        print(f"🔗 GET {voices_url}")
        print(f"➡️  Status: {response.status_code}")

        if response.ok:
            voices = response.json()
            print(f"✅ SUCCESS! Retrieved {len(voices)} voices.")

            matching_voice = next((voice for voice in voices if voice.get("ShortName") == voice_name), None)
            if matching_voice:
                print(f"   ✓ Desired voice '{voice_name}' is available.")
            else:
                print(f"   ⚠️ Voice '{voice_name}' not found in this region. Available sample:")
                for voice in voices[:5]:
                    print(f"     - {voice.get('ShortName')} ({voice.get('Locale')})")
        else:
            print("❌ Connection test failed.")
            print(f"   Response body: {response.text}")

    except requests.RequestException as exc:
        print("❌ ERROR: Network request failed.")
        print(f"   Details: {exc}")

    print("=" * 80)

if __name__ == "__main__":
    test_azure_tts()
