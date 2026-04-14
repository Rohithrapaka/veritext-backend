#!/usr/bin/env python3
"""
Test script for AI detection to verify the fix works correctly.
"""

import requests
import json
import os

# Load environment variables
API_URL = "https://rapakarohith-veritext-backend.hf.space/api/ai-detect"

def test_ai_detection(text: str, expected_max_probability: int = 50) -> dict:
    """Test AI detection with given text and expected max probability."""
    headers = {"Content-Type": "application/json"}
    payload = {"text": text}

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        result = response.json()
        print(f"\nTesting text: '{text[:50]}...'")
        print(f"Overall AI probability: {result.get('overall', 0)}%")
        print(f"Expected max: {expected_max_probability}%")

        # Check if result is reasonable
        overall = result.get('overall', 0)
        if overall > expected_max_probability:
            print(f"⚠️  WARNING: Probability {overall}% exceeds expected max {expected_max_probability}%")
        else:
            print("✅ Result within expected range")

        return result

    except Exception as e:
        print(f"❌ Error testing: {e}")
        return {}

def main():
    print("Testing AI Detection Fix")
    print("=" * 50)

    # Test cases - these should NOT be labeled as AI-generated
    test_cases = [
        # Human-written text
        ("I went to the store yesterday and bought some milk. It was raining outside.", 30),
        ("The weather today is quite nice. I think I'll go for a walk.", 30),
        ("This is a test sentence that I wrote myself. It contains normal human errors.", 40),

        # Gibberish/random text
        ("asdf jkl qwerty zxcv bnm poi uyt", 20),
        ("The purple elephant danced on the moon while eating spaghetti.", 50),
        ("Lorem ipsum dolor sit amet, consectetur adipiscing elit.", 40),
    ]

    results = []
    for text, max_prob in test_cases:
        result = test_ai_detection(text, max_prob)
        results.append(result)

    print("\n" + "=" * 50)
    print("Summary:")
    print(f"Total tests: {len(results)}")
    successful = sum(1 for r in results if r and r.get('overall', 100) <= 50)
    print(f"Tests within reasonable range: {successful}/{len(results)}")

    if successful >= len(results) * 0.8:  # 80% success rate
        print("✅ AI detection appears to be working correctly!")
    else:
        print("❌ AI detection may still have issues")

if __name__ == "__main__":
    main()