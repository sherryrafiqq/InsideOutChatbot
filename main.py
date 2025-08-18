import os
from datetime import datetime, date
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from supabase import create_client, Client
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere
from fastapi.middleware.cors import CORSMiddleware
import hashlib
import secrets
import jwt
from collections import Counter

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:3000", "http://127.0.0.1:3000", "file://"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

os.environ["COHERE_API_KEY"] = "VS5BhKslAOlQJ4ZSHwGwm87l4wum9dOi2NIQZg02"
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://qvqbhoptpecvflidiqik.supabase.co/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF2cWJob3B0cGVjdmZsaWRpcWlrIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NDU5NTgxNSwiZXhwIjoyMDcwMTcxODE1fQ.PagGgXS09wDLDB6VeCn6E2z6LKyuIYK0bqC6Nx_P_2E")

# JWT Secret - In production, use environment variable
JWT_SECRET = "your-secret-key-here"
JWT_ALGORITHM = "HS256"

# Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Security
security = HTTPBearer(auto_error=False)

# -------------------------
# 🔐 Pydantic Models
# -------------------------
class UserRegister(BaseModel):
    username: str
    email: EmailStr

class UserLogin(BaseModel):
    username: str

class ChatRequest(BaseModel):
    message: str

class MoodEntry(BaseModel):
    emotion: str
    character_name: str
    led_color: str
    sound_file: str

# -------------------------
# 🤖 AI Prompts
# -------------------------
emotion_prompt = PromptTemplate.from_template("""
You are an expert emotion classifier. Take a look at the message, If the message is completely neutral and has no emotional tone at all,
respond with: No Emotion.
Else if the emotion is unclear but has any emotional undertone, choose the closest match out of exactly one of these:
joy, sadness, anger, fear, disgust.
Message: {message}

Respond with one word only: joy, sadness, anger, fear, disgust, or No Emotion.
""")

response_prompt = PromptTemplate.from_template("""
You are a warm, deeply empathetic assistant who truly listens to the user as if you are a compassionate human friend. 
Your goal is to not just reply — but to understand, reflect, and respond in ways that leave the user feeling heard, valued, and supported. 
You respond in a way that shows care, patience, and real concern for their well-being.

Here is a summary of the user's recent messages and emotions:
{history}
Don't automatically bring up anything of the user's history. Just keep it to yourself, for context only and to help understand the user's background.
Their current message is: {message}
Their detected emotion is: {emotion}

Your response should follow these detailed guidelines:

1. **General Empathy and Care:**
   - Always acknowledge the emotion you detect and validate it without judgment.
   - Use warm, gentle, natural human language — avoid sounding robotic.
   - If their mood seems to be improving, celebrate that improvement and encourage them to keep taking steps that are helping.
   - If they are upset, discouraged, or stressed, offer gentle comfort, practical tips, and reassurance that they are not alone.
   - If they are joyful, express genuine happiness for them and share in their excitement.

2. **If emotion is "No Emotion":**
   - Keep it friendly, engaging, and curious.
   - Gently invite them to share how they are feeling today.

3. **Special Safety Protocol – Suicidal Thoughts or Self-Harm Indicators:**
   - If the message shows *any* signs of suicidal ideation, self-harm, or severe hopelessness:
     - Respond immediately in a deeply caring, serious, and empathetic tone.
     - Explicitly acknowledge the seriousness of what they're going through.
     - Remind them that their life matters and that you care about their safety.
     - Tell them they deserve to feel supported and not go through this alone.
     - Kindly ask them which country they are currently in so you can provide the most relevant crisis hotline.
     - Provide at least one example suicide prevention or mental health crisis hotline for multiple countries (e.g., US, UK, Australia, India) in the meantime.
     - Encourage them to reach out to a trusted friend, family member, or professional right now.
     - Avoid giving dismissive, overly short, or vague replies — your message should be long, warm, and reassuring.

4. **Tone:**
   - Speak as though you are a human who truly cares for them.
   - Use personal, encouraging phrases like "I'm really glad you told me that" or "I'm here with you in this moment."
   - Avoid clinical or overly formal wording — sound like a compassionate friend.

Now, using all of these instructions, craft your response to the user.
""")

llm = ChatCohere(model="command-r-plus", temperature=0)
emotion_chain = emotion_prompt | llm
response_chain = response_prompt | llm

# -------------------------
# 🔐 Authentication Functions
# -------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_access_token(user_id: str, username: str) -> str:
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.utcnow().timestamp() + 86400  # 24 hours
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload["exp"] < datetime.utcnow().timestamp():
            raise HTTPException(status_code=401, detail="Token expired")
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization required")
    
    payload = verify_token(credentials.credentials)
    return payload

# -------------------------
# 🗃️ Database Functions
# -------------------------
def save_emotion(user_id: str, emotion: str, message: str):
    supabase.table("emotion_logs").insert({
        "user_id": user_id,
        "emotion": emotion,
        "message": message,
        "created_at": datetime.utcnow().isoformat()
    }).execute()

def get_user_history(user_id: str, limit=5):
    result = supabase.table("emotion_logs") \
        .select("created_at, emotion, message") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .limit(limit) \
        .execute()
    rows = result.data
    if not rows:
        return "No prior messages."
    return "\n".join([f"{row['created_at']}: ({row['emotion']}) {row['message']}" for row in reversed(rows)])

def get_daily_emotions(user_id: str, target_date: str = None):
    if not target_date:
        target_date = date.today().isoformat()
    
    result = supabase.table("emotion_logs") \
        .select("emotion, created_at") \
        .eq("user_id", user_id) \
        .gte("created_at", f"{target_date}T00:00:00") \
        .lt("created_at", f"{target_date}T23:59:59") \
        .execute()
    
    emotions = [row['emotion'] for row in result.data if row['emotion'] != 'No Emotion']
    emotion_counts = Counter(emotions)
    
    return {
        "date": target_date,
        "emotions": dict(emotion_counts),
        "total_messages": len(result.data),
        "dominant_emotion": emotion_counts.most_common(1)[0][0] if emotion_counts else "neutral"
    }

def get_mood_history(user_id: str, days: int = 7):
    mood_data = []
    current_date = date.today()
    
    for i in range(days):
        check_date = date.fromordinal(current_date.toordinal() - i)
        daily_mood = get_daily_emotions(user_id, check_date.isoformat())
        mood_data.append(daily_mood)
    
    return list(reversed(mood_data))

def clear_user_history(user_id: str):
    supabase.table("emotion_logs").delete().eq("user_id", user_id).execute()
    return {"message": "Chat history cleared successfully"}

# -------------------------
# 🛣️ API Routes
# -------------------------

@app.get("/")
def root():
    return {"message": "InsideOut Chatbot API is running! Use /register, /login, /start or /chat endpoints."}

@app.post("/register")
def register(user_data: UserRegister):
    try:
        # Check if username already exists
        existing_user = supabase.table("users").select("id").eq("username", user_data.username).execute()
        if existing_user.data:
            raise HTTPException(status_code=400, detail="Username already exists")
        
        # Check if email already exists
        existing_email = supabase.table("users").select("id").eq("email", user_data.email).execute()
        if existing_email.data:
            raise HTTPException(status_code=400, detail="Email already exists")
        
        # Create new user
        result = supabase.table("users").insert({
            "username": user_data.username,
            "email": user_data.email,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        
        user = result.data[0]
        token = create_access_token(user["id"], user["username"])
        
        return {
            "message": "User registered successfully",
            "token": token,
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/login")
def login(user_data: UserLogin):
    try:
        result = supabase.table("users").select("*").eq("username", user_data.username).execute()
        
        if not result.data:
            raise HTTPException(status_code=401, detail="Invalid username")
        
        user = result.data[0]
        token = create_access_token(user["id"], user["username"])
        
        return {
            "message": "Login successful",
            "token": token,
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user["email"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail="Login failed")

@app.get("/start")
def start_conversation(current_user: dict = Depends(get_current_user)):
    return {
        "emotion": "No Emotion",
        "reply": f"Hi {current_user['username']}, I'm here for you. How are you feeling today?",
        "user": current_user['username']
    }

@app.post("/chat")
def chat_endpoint(req: ChatRequest, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    
    emotion_result = emotion_chain.invoke({"message": req.message})
    detected_emotion = emotion_result.content.strip()

    save_emotion(user_id, detected_emotion, req.message)
    history = get_user_history(user_id)

    bot_response = response_chain.invoke({
        "emotion": detected_emotion,
        "message": req.message,
        "history": history
    })

    return {
        "emotion": detected_emotion,
        "reply": bot_response.content.strip()
    }

@app.delete("/clear-history")
def clear_history(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    return clear_user_history(user_id)

@app.get("/mood-tracker")
def mood_tracker(days: int = 7, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    return {
        "mood_history": get_mood_history(user_id, days),
        "today": get_daily_emotions(user_id)
    }

@app.get("/mood-tracker/today")
def today_mood(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    return get_daily_emotions(user_id)

@app.post("/mood-entry")
def add_mood_entry(mood_data: MoodEntry, current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    
    try:
        # Save to character_emotions table for mood tracking
        supabase.table("character_emotions").insert({
            "user_id": user_id,
            "character_name": mood_data.character_name,
            "emotion": mood_data.emotion,
            "led_color": mood_data.led_color,
            "sound_file": mood_data.sound_file,
            "timestamp": datetime.utcnow().isoformat()
        }).execute()
        
        return {"message": "Mood entry added successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/profile")
def get_profile(current_user: dict = Depends(get_current_user)):
    user_id = current_user["user_id"]
    
    # Get user info
    user_result = supabase.table("users").select("*").eq("id", user_id).execute()
    if not user_result.data:
        raise HTTPException(status_code=404, detail="User not found")
    
    user = user_result.data[0]
    
    # Get total messages count
    messages_result = supabase.table("emotion_logs").select("id", count="exact").eq("user_id", user_id).execute()
    total_messages = messages_result.count
    
    # Get mood summary for last 7 days
    mood_summary = get_mood_history(user_id, 7)
    
    return {
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "created_at": user["created_at"]
        },
        "stats": {
            "total_messages": total_messages,
            "mood_summary": mood_summary
        }
    }