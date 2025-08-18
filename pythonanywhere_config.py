# pythonanywhere_config.py - PythonAnywhere Deployment Configuration
"""
This file contains configuration settings specific to PythonAnywhere deployment.
Copy these settings to your PythonAnywhere environment variables or .env file.
"""

# Environment Variables for PythonAnywhere
ENVIRONMENT_VARIABLES = {
    # AI Configuration
    "COHERE_API_KEY": "your-cohere-api-key-here",
    
    # Database Configuration
    "SUPABASE_URL": "https://qvqbhoptpecvflidiqik.supabase.co/",
    "SUPABASE_KEY": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF2cWJob3B0cGVjdmZsaWRpcWlrIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDU5NTgxNSwiZXhwIjoyMDcwMTcxODE1fQ.PagGgXS09wDLDB6VeCn6E2z6LKyuIYK0bqC6Nx_P_2E",
    
    # Security Configuration
    "JWT_SECRET": "your-super-secret-jwt-key-change-me-in-production",
    
    # Application Configuration
    "DEBUG": "False",  # Set to "True" for development
    "PYTHONANYWHERE_SITE": "yourusername.pythonanywhere.com"
}

# PythonAnywhere Web App Configuration
WEB_APP_CONFIG = {
    "source_code": "/home/yourusername/IoT-Sentiment-Analysis",
    "working_directory": "/home/yourusername/IoT-Sentiment-Analysis",
    "wsgi_file": "/home/yourusername/IoT-Sentiment-Analysis/wsgi.py",
    "python_version": "3.9",  # or your preferred version
    "virtual_environment": "/home/yourusername/.virtualenvs/insideout_env"
}

# Database Tables Required
REQUIRED_TABLES = [
    "users",
    "emotion_logs", 
    "character_emotions"
]

# API Endpoints
API_ENDPOINTS = [
    "GET / - Health check",
    "GET /health - Detailed health check",
    "POST /register - User registration",
    "POST /login - User login",
    "GET /start - Start conversation",
    "POST /chat - Main chat endpoint",
    "DELETE /clear-history - Clear chat history",
    "GET /mood-tracker - Get mood tracking data",
    "GET /mood-tracker/today - Get today's mood",
    "POST /mood-entry - Add mood entry",
    "GET /profile - Get user profile"
]

# Deployment Checklist
DEPLOYMENT_CHECKLIST = [
    "1. Upload all files to PythonAnywhere",
    "2. Create virtual environment and install requirements",
    "3. Set environment variables in PythonAnywhere",
    "4. Configure web app to use wsgi.py",
    "5. Set up database tables in Supabase",
    "6. Test API endpoints",
    "7. Configure CORS for your Flutter app domain",
    "8. Set up SSL certificate (if needed)"
]

def print_config():
    """Print configuration for easy copying"""
    print("=== PythonAnywhere Environment Variables ===")
    for key, value in ENVIRONMENT_VARIABLES.items():
        print(f"{key}={value}")
    
    print("\n=== Web App Configuration ===")
    for key, value in WEB_APP_CONFIG.items():
        print(f"{key}: {value}")
    
    print("\n=== Required Database Tables ===")
    for table in REQUIRED_TABLES:
        print(f"- {table}")
    
    print("\n=== API Endpoints ===")
    for endpoint in API_ENDPOINTS:
        print(f"- {endpoint}")
    
    print("\n=== Deployment Checklist ===")
    for item in DEPLOYMENT_CHECKLIST:
        print(item)

if __name__ == "__main__":
    print_config() 