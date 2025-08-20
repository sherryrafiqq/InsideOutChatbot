# InsideOut Chatbot API (Simplified)

A simplified version of the InsideOut Chatbot API that focuses solely on the chatbot functionality without user authentication, registration, or database storage.

## 🚀 What's Changed

This version has been simplified by removing all authentication and database-related features:

### ❌ Removed Features
- User registration and login
- JWT token authentication
- User profiles and statistics
- Mood tracking and history
- Database storage (Supabase)
- User-specific chat history
- Profile management

### ✅ Kept Features
- AI-powered emotion detection
- Empathetic chatbot responses
- Health checks and monitoring
- Rate limiting and security
- CORS configuration
- Error handling and logging

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "IoT Sentiment Analysis"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env and add your COHERE_API_KEY
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# AI Configuration
COHERE_API_KEY=your_cohere_api_key_here

# Application Configuration
DEBUG=False
PORT=8000

# Platform Detection (optional)
PYTHONANYWHERE_SITE=
```

### Required API Keys

- **COHERE_API_KEY**: Get your API key from [Cohere](https://cohere.ai/)

## 📡 API Endpoints

### Health Check
- `GET /` - Root endpoint with API status
- `GET /health` - Detailed health check
- `GET /health/detailed` - Comprehensive health status

### Chatbot
- `GET /start` - Start a new conversation
- `POST /chat` - Send a message and get AI response
- `POST /test-chat` - Test endpoint without authentication

### Testing
- `GET /test` - Serve the test HTML page

## 💬 Usage Examples

### Start a Conversation
```bash
curl -X GET "http://localhost:8000/start"
```

### Send a Message
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! I am feeling happy today!"}'
```

### Test Chat
```bash
curl -X POST "http://localhost:8000/test-chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "This is a test message"}'
```

## 🧪 Testing

### Web Interface
Visit `http://localhost:8000/test` to access the web-based testing interface.

### Command Line Testing
```bash
# Test against local server
python test_chatbot_only.py

# Test against deployed server
python test_chatbot_only.py https://your-deployment-url.com
```

### Deployment Testing
```bash
python test_deployment.py
```

## 🚀 Deployment

### Railway
The application is configured for Railway deployment with the `railway.json` file.

### PythonAnywhere
Update the `BASE_URL` in `test_deployment.py` and deploy using PythonAnywhere's WSGI configuration.

### Other Platforms
The application uses standard FastAPI and should work on any platform that supports Python web applications.

## 📊 Response Format

All API responses follow this standard format:

```json
{
  "success": true,
  "message": "Chat response generated",
  "data": {
    "emotion": "joy",
    "reply": "I'm so glad you're feeling happy today!"
  },
  "error": null,
  "timestamp": "2024-01-01T12:00:00.000000"
}
```

## 🔒 Security Features

- Rate limiting (60 requests per minute per IP)
- Input validation and sanitization
- Error handling without information leakage
- CORS configuration for web applications

## 🐛 Troubleshooting

### Common Issues

1. **AI Service Unavailable**
   - Check your `COHERE_API_KEY` is valid
   - Verify internet connectivity
   - Check Cohere service status

2. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

3. **Port Already in Use**
   - Change the `PORT` environment variable
   - Kill existing processes using the port

### Logs
The application logs to stdout and optionally to `/tmp/insideout_api.log` on supported platforms.

## 📝 License

This project is part of the IoT Sentiment Analysis system.

## 🤝 Contributing

This is a simplified version focused on core chatbot functionality. For the full version with authentication and database features, please refer to the original repository.
