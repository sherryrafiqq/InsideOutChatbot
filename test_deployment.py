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
TEST_USERNAME = "testuser"
TEST_EMAIL = "test@example.com"

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
    
    # Test 3: User registration
    print("3. Testing User Registration")
    registration_data = {
        "username": TEST_USERNAME,
        "email": TEST_EMAIL
    }
    if test_endpoint("/register", method="POST", data=registration_data):
        tests_passed += 1
    total_tests += 1
    print()
    
    # Test 4: User login
    print("4. Testing User Login")
    login_data = {
        "username": TEST_USERNAME
    }
    login_response = requests.post(f"{BASE_URL}/login", json=login_data)
    if login_response.status_code == 200:
        try:
            login_result = login_response.json()
            if login_result.get('success') and login_result.get('token'):
                token = login_result['token']
                headers = {"Authorization": f"Bearer {token}"}
                print("✅ Login successful, token obtained")
                tests_passed += 1
            else:
                print("❌ Login failed - no token received")
        except:
            print("❌ Login failed - invalid response")
    else:
        print(f"❌ Login failed - Status: {login_response.status_code}")
    total_tests += 1
    print()
    
    # Test 5: Start conversation (requires authentication)
    print("5. Testing Start Conversation")
    if 'headers' in locals():
        if test_endpoint("/start", headers=headers):
            tests_passed += 1
    else:
        print("❌ Skipped - no authentication token")
    total_tests += 1
    print()
    
    # Test 6: Chat endpoint (requires authentication)
    print("6. Testing Chat Endpoint")
    if 'headers' in locals():
        chat_data = {
            "message": "Hello, I'm feeling happy today!"
        }
        if test_endpoint("/chat", method="POST", data=chat_data, headers=headers):
            tests_passed += 1
    else:
        print("❌ Skipped - no authentication token")
    total_tests += 1
    print()
    
    # Test 7: Profile endpoint (requires authentication)
    print("7. Testing Profile Endpoint")
    if 'headers' in locals():
        if test_endpoint("/profile", headers=headers):
            tests_passed += 1
    else:
        print("❌ Skipped - no authentication token")
    total_tests += 1
    print()
    
    # Test 8: Mood tracker (requires authentication)
    print("8. Testing Mood Tracker")
    if 'headers' in locals():
        if test_endpoint("/mood-tracker", headers=headers):
            tests_passed += 1
    else:
        print("❌ Skipped - no authentication token")
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