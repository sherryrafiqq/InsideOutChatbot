#!/usr/bin/env python3
"""
Test script for PythonAnywhere deployment
Run this script to verify your API is working correctly
"""

import requests
import json
import sys
from datetime import datetime

# Configuration
BASE_URL = "https://yourusername.pythonanywhere.com"  # Replace with your actual URL

def test_endpoint(endpoint, method="GET", data=None, headers=None):
    """Test a single endpoint"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, json=data, headers=headers)
        elif method == "DELETE":
            response = requests.delete(url, headers=headers)
        else:
            print(f"❌ Unsupported method: {method}")
            return False
        
        print(f"✅ {method} {endpoint} - Status: {response.status_code}")
        
        if response.status_code < 400:
            try:
                result = response.json()
                if isinstance(result, dict) and result.get('success'):
                    print(f"   Response: {result.get('message', 'Success')}")
                else:
                    print(f"   Response: {result}")
            except:
                print(f"   Response: {response.text[:100]}...")
        else:
            print(f"   Error: {response.text}")
        
        return response.status_code < 400
        
    except Exception as e:
        print(f"❌ {method} {endpoint} - Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing InsideOut Chatbot API Deployment")
    print("=" * 50)
    print(f"Base URL: {BASE_URL}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Test results
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Health check
    print("1. Testing Health Check")
    if test_endpoint("/health"):
        tests_passed += 1
    total_tests += 1
    print()
    
    # Test 2: Root endpoint
    print("2. Testing Root Endpoint")
    if test_endpoint("/"):
        tests_passed += 1
    total_tests += 1
    print()
    
    # Test 3: Start conversation
    print("3. Testing Start Conversation")
    if test_endpoint("/start"):
        tests_passed += 1
    total_tests += 1
    print()
    
    # Test 4: Chat endpoint
    print("4. Testing Chat Endpoint")
    chat_data = {
        "message": "Hello, I'm feeling happy today!"
    }
    if test_endpoint("/chat", method="POST", data=chat_data):
        tests_passed += 1
    total_tests += 1
    print()
    
    # Test 5: Test chat endpoint
    print("5. Testing Test Chat Endpoint")
    test_chat_data = {
        "message": "This is a test message"
    }
    if test_endpoint("/test-chat", method="POST", data=test_chat_data):
        tests_passed += 1
    total_tests += 1
    print()
    
    # Summary
    print("=" * 50)
    print("📊 Test Summary")
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Success Rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your deployment is working correctly.")
        return 0
    elif tests_passed >= total_tests * 0.7:
        print("⚠️  Most tests passed. Check the failed tests above.")
        return 1
    else:
        print("❌ Many tests failed. Please check your deployment configuration.")
        return 1

if __name__ == "__main__":
    # Check if BASE_URL is configured
    if BASE_URL == "https://yourusername.pythonanywhere.com":
        print("❌ Please update the BASE_URL in this script with your actual PythonAnywhere URL")
        print("   Edit the BASE_URL variable at the top of this file")
        sys.exit(1)
    
    exit_code = main()
    sys.exit(exit_code) 