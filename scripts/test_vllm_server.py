#!/usr/bin/env python3
"""
Test script to verify vLLM server is working correctly.

Usage:
    python scripts/test_vllm_server.py [MODEL_SIZE] [--base-url http://localhost:8000]

Examples:
    python scripts/test_vllm_server.py 8
    python scripts/test_vllm_server.py 70
    python scripts/test_vllm_server.py 405
    python scripts/test_vllm_server.py 8 --base-url http://remote-server:8000
"""

import argparse
import requests
import json
import sys


# Model size to model name mapping
MODEL_NAMES = {
    "8": "meta-llama/Llama-3.1-8B",
    "8B": "meta-llama/Llama-3.1-8B",
    "70": "meta-llama/Meta-Llama-3.1-70B",
    "70B": "meta-llama/Meta-Llama-3.1-70B",
    "405": "meta-llama/Meta-Llama-3.1-405B",
    "405B": "meta-llama/Meta-Llama-3.1-405B",
}


def get_model_name(model_size):
    """
    Convert model size to full model name.

    Args:
        model_size: Model size (8, 70, or 405)

    Returns:
        Full model name
    """
    model_size_upper = model_size.upper()
    if model_size_upper in MODEL_NAMES:
        return MODEL_NAMES[model_size_upper]
    elif model_size in MODEL_NAMES:
        return MODEL_NAMES[model_size]
    else:
        raise ValueError(f"Invalid model size '{model_size}'. Valid options: 8, 70, 405")


def test_vllm_server(base_url="http://localhost:8000", model_size="405"):
    """
    Test vLLM server with a simple completion request.

    Args:
        base_url: URL of vLLM server
        model_size: Model size (8, 70, or 405)

    Returns:
        True if test passes, False otherwise
    """
    try:
        model_name = get_model_name(model_size)
    except ValueError as e:
        print(f"✗ {e}")
        return False

    print(f"Testing vLLM server at {base_url}...")
    print(f"Model: {model_name}")
    print("=" * 80)

    # Test basic completion
    try:
        response = requests.post(
            f"{base_url}/v1/completions",
            json={
                "model": model_name,
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
    parser = argparse.ArgumentParser(
        description="Test vLLM server functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_vllm_server.py 8
  python scripts/test_vllm_server.py 70
  python scripts/test_vllm_server.py 405
  python scripts/test_vllm_server.py 8 --base-url http://remote-server:8000
        """
    )
    parser.add_argument(
        "model_size",
        nargs="?",
        default="405",
        help="Model size: 8, 70, or 405 (default: 405)"
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of vLLM server (default: http://localhost:8000)"
    )
    args = parser.parse_args()

    success = test_vllm_server(args.base_url, args.model_size)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
