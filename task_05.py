<<<<<<< HEAD
import os
from google import genai
from dotenv import load_dotenv

# Load API Key
load_dotenv(".myenv")
api_key = os.getenv("GEMINI_API_KEY")

# Gemini Client
client = genai.Client(api_key=api_key)

system_prompt = """
You are an empathetic emotional-support assistant.
Your goal is to comfort individuals dealing with stress, anxiety, sadness, or emotional problems.
Use simple, warm, human-like language. Avoid medical advice.
Always validate emotions, show understanding, and offer gentle support.
"""

def get_response(user_message):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user_message,
        config={"system_instruction": system_prompt}
    )
    return response.text

def start_chat():
    print("🌿 Welcome to the Emotional Wellness Support Chatbot")
    print("Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Bot: Take care! I'm always here if you need support. 💚")
            break

        bot_reply = get_response(user_input)
        print("Bot:", bot_reply)

if __name__ == "__main__":
    start_chat()
=======
import os
from google import genai
from dotenv import load_dotenv

# Load API Key
load_dotenv(".myenv")
api_key = os.getenv("GEMINI_API_KEY")

# Gemini Client
client = genai.Client(api_key=api_key)

system_prompt = """
You are an empathetic emotional-support assistant.
Your goal is to comfort individuals dealing with stress, anxiety, sadness, or emotional problems.
Use simple, warm, human-like language. Avoid medical advice.
Always validate emotions, show understanding, and offer gentle support.
"""

def get_response(user_message):
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user_message,
        config={"system_instruction": system_prompt}
    )
    return response.text

def start_chat():
    print("🌿 Welcome to the Emotional Wellness Support Chatbot")
    print("Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Bot: Take care! I'm always here if you need support. 💚")
            break

        bot_reply = get_response(user_input)
        print("Bot:", bot_reply)

if __name__ == "__main__":
    start_chat()
>>>>>>> f45f1a62ece1e1f34b6fa93c3056c562f6c7a797
