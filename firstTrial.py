import os
import sqlite3
from datetime import datetime
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere

# -------------------------
# 🔐 Set up Cohere API Key
# -------------------------
os.environ["COHERE_API_KEY"] = "VS5BhKslAOlQJ4ZSHwGwm87l4wum9dOi2NIQZg02"

# -------------------------
# 🧠 Create Prompt Templates
# -------------------------
emotion_prompt = PromptTemplate.from_template("""
You are an expert emotion classifier.
Classify the main emotion in the given message into exactly one of these:
joy, sadness, anger, fear, disgust.

If the message is completely neutral and has no emotional tone at all,
respond with: No Emotion.

If the emotion is unclear but has any emotional undertone, choose the closest match.

Message: {message}

Respond with one word only: joy, sadness, anger, fear, disgust, or No Emotion.
""")

response_prompt = PromptTemplate.from_template("""
You are a warm, empathetic assistant and a great listener. 
You speak in a supportive, non-judgmental, and encouraging tone.

Here is a summary of their past emotions and messages:
{history}

Their current message is: {message}
Their current detected emotion is: {emotion}

Respond in a way that:
- Acknowledges their feelings and validates their experience.
- Encourages them when their mood is improving.
- Comforts them and offers gentle, practical suggestions if they’re upset.
- Celebrates with them if they are feeling joyful.
- If they display signs of suicidal thoughts or self-harm, be extremely supportive, ask helpfully where is their residential area, and provide helpline numbers for their country.
- If "No Emotion" is detected, respond neutrally but in a friendly and engaging way to keep the conversation going.

Keep your answer concise, human-like, and caring.
""")

llm = ChatCohere(model="command-r-plus", temperature=0)

# -------------------------
# 🔗 Create Chains
# -------------------------
emotion_chain = emotion_prompt | llm
response_chain = response_prompt | llm

# -------------------------
# 🗃️ Setup SQLite DB
# -------------------------
def setup_db():
    conn = sqlite3.connect("emotion_log.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS emotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            timestamp TEXT,
            emotion TEXT,
            message TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_emotion(user_id, emotion, message):
    conn = sqlite3.connect("emotion_log.db")
    cursor = conn.cursor()
    timestamp = datetime.now().isoformat(timespec='seconds')
    cursor.execute("""
        INSERT INTO emotions (user_id, timestamp, emotion, message) 
        VALUES (?, ?, ?, ?)
    """, (user_id, timestamp, emotion, message))
    conn.commit()
    conn.close()

def get_user_history(user_id, limit=5):
    conn = sqlite3.connect("emotion_log.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT timestamp, emotion, message FROM emotions 
        WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?
    """, (user_id, limit))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return "No prior messages."
    
    history = "\n".join(
        [f"{row[0]}: ({row[1]}) {row[2]}" for row in reversed(rows)]
    )
    return history

# -------------------------
# 🗨️ Start Chatbot
# -------------------------
def start_chat():
    setup_db()

    print("💬 Welcome to InsideOut Chatbot!")
    user_id = input("Enter your username: ").strip()
    if not user_id:
        print("Username is required. Exiting.")
        return

    print(f"Hello {user_id}! How are you feeling today?")
    
    while True:
        user_message = input("You: ").strip()
        if user_message.lower() in ["exit", "quit"]:
            print("Bot: Take care! See you next time!")
            break

        # Detect emotion
        emotion_result = emotion_chain.invoke({"message": user_message})
        detected_emotion = emotion_result.content.strip()

        # If no valid emotion, set as "No Emotion"
        if detected_emotion.lower() not in ["joy", "sadness", "anger", "fear", "disgust"]:
            emotion = "No Emotion"
        else:
            emotion = detected_emotion

        # Save emotion with user
        save_emotion(user_id, emotion, user_message)

        # Get user history
        history = get_user_history(user_id)

        # Generate response with history context
        bot_response = response_chain.invoke({
            "emotion": emotion,
            "message": user_message,
            "history": history
        })

        print("Bot:", bot_response.content.strip())

# -------------------------
# 🚀 Run the chatbot
# -------------------------
if __name__ == "__main__":
    start_chat()
