#!/usr/bin/env python3
"""
Test script to verify vLLM server is working correctly.

Usage:
    python scripts/test_vllm_server.py [--base-url http://localhost:8000]
"""

import argparse
import requests
import json
import sys


def test_vllm_server(base_url="http://localhost:8000"):
    """
    Test vLLM server with a simple completion request.

    Args:
        base_url: URL of vLLM server

    Returns:
        True if test passes, False otherwise
    """
    print(f"Testing vLLM server at {base_url}...")
    print("=" * 80)

    # Test basic completion
    try:
        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": "meta-llama/Meta-Llama-3.1-405B",  # Adjust based on your deployment
                "prompt": "Question: Is the sky blue?\nClaim: The sky is blue.\nI think this claim is ",
                "max_tokens": 1,
                "logprobs": 20,
                "temperature": 0.0
            },
            timeout=60
        )

        if response.status_code != 200:
            print(f"✗ Request failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False

        result = response.json()
        print("✓ Server responded successfully")
        print("\nResponse structure:")
        print(json.dumps(result, indent=2))

        # Verify logprobs are present
        if "choices" not in result:
            print("✗ Response missing 'choices' field")
            return False

        choice = result["choices"][0]
        if "logprobs" not in choice or choice["logprobs"] is None:
            print("✗ Response missing logprobs")
            return False

        print("\n✓ Logprobs are present")
        print(f"  Completion: {choice['text']}")
        print(f"  Top logprobs: {choice['logprobs'].get('top_logprobs', [])[:1]}")

        print("\n" + "=" * 80)
        print("✓ vLLM server is working correctly!")
        return True

    except requests.exceptions.Timeout:
        print("✗ Request timed out. Is the server running?")
        return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Could not connect to {base_url}. Is the server running?")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test vLLM server functionality")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of vLLM server (default: http://localhost:8000)"
    )
    args = parser.parse_args()

    success = test_vllm_server(args.base_url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
