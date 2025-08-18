# InsideOut Chatbot API

An empathetic AI chatbot for emotion support built with FastAPI, Cohere, and Supabase.

## Features

- 🤖 AI-powered emotional support chatbot
- 🗃️ User authentication and session management
- 📊 Mood tracking and analytics
- 🔐 Secure JWT authentication
- 🌐 CORS-enabled for web applications
- 📱 Compatible with Flutter and web apps

## Environment Variables

Create a `.env` file in your project root with the following variables:

```env
# AI Configuration
COHERE_API_KEY=your_cohere_api_key_here

# Database Configuration (Supabase)
SUPABASE_URL=https://your-project.supabase.co/
SUPABASE_KEY=your_supabase_anon_key_here

# Security
JWT_SECRET=your_super_secret_jwt_key_change_me_in_production

# Application Configuration
DEBUG=False
PORT=8000
```

## Local Development

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   ```bash
   cp env.example .env
   # Edit .env with your actual values
   ```

3. **Run the development server:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Access the API:**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Test Page: http://localhost:8000/test

## Deployment on Render

### Option 1: Using render.yaml (Recommended)

1. **Push your code to GitHub**
2. **Connect your repository to Render**
3. **Render will automatically detect the `render.yaml` file**
4. **Set your environment variables in the Render dashboard:**
   - `COHERE_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_KEY`
   - `JWT_SECRET` (auto-generated)
   - `DEBUG=false`
   - `RENDER=true`

### Option 2: Manual Setup

1. **Create a new Web Service on Render**
2. **Connect your GitHub repository**
3. **Configure the service:**
   - **Environment:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. **Add environment variables in the dashboard**

## API Endpoints

### Authentication
- `POST /register` - User registration
- `POST /login` - User login

### Chat
- `GET /start` - Start conversation
- `POST /chat` - Send message to chatbot

### Mood Tracking
- `GET /mood-tracker` - Get mood history
- `POST /mood-entry` - Add mood entry

### Health & Info
- `GET /` - API information
- `GET /health` - Health check
- `GET /test` - Test page

## Database Setup

Ensure your Supabase database has the following tables:

### users
```sql
CREATE TABLE users (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  username VARCHAR(50) UNIQUE NOT NULL,
  email VARCHAR(255) UNIQUE NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### emotion_logs
```sql
CREATE TABLE emotion_logs (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  emotion VARCHAR(50) NOT NULL,
  message TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### character_emotions
```sql
CREATE TABLE character_emotions (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES users(id),
  character_name VARCHAR(100) NOT NULL,
  emotion VARCHAR(50) NOT NULL,
  led_color VARCHAR(7) NOT NULL,
  sound_file VARCHAR(255) NOT NULL,
  timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Security Notes

- Never commit your `.env` file to version control
- Use strong, unique JWT secrets in production
- Keep your API keys secure
- Enable HTTPS in production
- Regularly rotate your secrets

## Support

For issues and questions, please check the API documentation at `/docs` when running locally.
