#!/usr/bin/env python3
"""
Simple test script for the chatbot-only version
Tests the core chatbot functionality without authentication
"""

import requests
import json
import sys

# Configuration
BASE_URL = "http://localhost:8000"  # Change this to your actual URL

def test_chatbot():
    """Test the chatbot functionality"""
    print("🤖 Testing InsideOut Chatbot (No Authentication)")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing Health Check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data.get('message', 'Success')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Start conversation
    print("2. Testing Start Conversation...")
    try:
        response = requests.get(f"{BASE_URL}/start")
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Start conversation: {data['data']['reply'][:50]}...")
            else:
                print(f"❌ Start conversation failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Start conversation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Start conversation error: {e}")
        return False
    
    # Test 3: Chat endpoint
    print("3. Testing Chat Endpoint...")
    try:
        chat_data = {"message": "Hello! I'm feeling happy today!"}
        response = requests.post(f"{BASE_URL}/chat", json=chat_data)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                emotion = data['data']['emotion']
                reply = data['data']['reply'][:50]
                print(f"✅ Chat response: Emotion={emotion}, Reply={reply}...")
            else:
                print(f"❌ Chat failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Chat failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return False
    
    # Test 4: Test chat endpoint
    print("4. Testing Test Chat Endpoint...")
    try:
        test_data = {"message": "This is a test message"}
        response = requests.post(f"{BASE_URL}/test-chat", json=test_data)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Test chat: {data.get('message', 'Success')}")
            else:
                print(f"❌ Test chat failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Test chat failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Test chat error: {e}")
        return False
    
    print("\n🎉 All tests passed! The chatbot is working correctly.")
    return True

if __name__ == "__main__":
    if len(sys.argv) > 1:
        BASE_URL = sys.argv[1]
    
    print(f"Testing against: {BASE_URL}")
    success = test_chatbot()
    sys.exit(0 if success else 1)
