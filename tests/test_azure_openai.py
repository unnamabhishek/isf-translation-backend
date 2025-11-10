#!/usr/bin/env python3
"""
Test LiteLLM-hosted Translation
Run this to verify the LiteLLM gateway is reachable with your credentials.
"""

import os
import sys
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv(".env.local")


def _masked(value: Optional[str]) -> str:
    if not value:
        return "NOT SET"
    return f"{value[:8]}…"


def test_azure_openai():
    """Test LiteLLM / GPT translation"""

    print("=" * 80)
    print("Testing LiteLLM Translation")
    print("=" * 80)

    # Get credentials
    api_key = os.getenv("LITELLM_API_KEY")
    api_base = os.getenv("LITELLM_API_BASE")
    model_name = os.getenv("LITELLM_MODEL", "gpt-4.1")

    print(f"LiteLLM Base: {api_base}")
    print(f"Model: {model_name}")
    print(f"API Key: {_masked(api_key)}")
    print()

    if not api_key or not api_base:
        print("❌ ERROR: LITELLM_API_KEY or LITELLM_API_BASE not set in .env.local", file=sys.stderr)
        return

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=api_base.rstrip("/"),
        )

        # Test English to Hindi translation
        test_text = "Hello, how are you today?"

        print(f"📝 Test English text: {test_text}")
        print(f"🔄 Translating to Hindi...")
        print()

        # Make request
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a translator. Translate the following English text to Hindi. Provide ONLY the Hindi translation, nothing else."
                },
                {
                    "role": "user",
                    "content": test_text
                }
            ],
            temperature=0.3,
            max_tokens=100
        )

        hindi_translation = response.choices[0].message.content.strip()

        print("-" * 80)
        print("✅ SUCCESS! Translation completed:")
        print(f"   English: {test_text}")
        print(f"   Hindi: {hindi_translation}")
        print()
        print("✓ LiteLLM gateway responded correctly!")
        print(f"✓ Model used: {response.model}")
        print(f"✓ Tokens used: {response.usage.total_tokens}")

    except Exception as e:
        print("❌ ERROR occurred:")
        print(f"   Error Type: {type(e).__name__}")
        print(f"   Error Message: {str(e)}")
        print()

        # Show more details
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        print()

        print("Common issues:")
        print("  - Check if LITELLM_API_BASE points to the correct /v1 endpoint")
        print("  - Verify the API key is valid and not expired")
        print("  - Confirm the upstream model name matches LITELLM_MODEL")
        print("  - Test connectivity: curl -I", api_base)

    print("=" * 80)


if __name__ == "__main__":
    test_azure_openai()
