# Imported needed libararies 
import time
import os
from google import genai
from google.genai.errors import ServerError
from dotenv import load_dotenv

# Load API key from .myenv
load_dotenv(".myenv")
api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini client
client = genai.Client(api_key=api_key)

# System instruction for friendly & safe health assistant
SYSTEM_INSTRUCTION = (
    "You are a friendly, knowledgeable, and helpful general health assistant. "
    "Provide clear, easy-to-understand answers about common health questions. "
    "CRITICAL SAFETY: Never give personalized medical advice or diagnosis. "
    "If asked about treatment, respond: 'I am an AI assistant and cannot provide medical advice. "
    "Please consult a qualified healthcare professional.'"
)

MODEL_NAME = "gemini-2.5-flash"

print("--- Gemini Health Chatbot Initialized ---")
print("Type Quit or Exit to End the chat.")

# Interactive chat loop
while True:
    user_input = input("\nPatient Query: ").strip()
    if user_input.lower() in ["quit", "exit", "bye"]:
        print("Assistant: Stay healthy! Goodbye!")
        break
    if not user_input:
        continue

    prompt = f"{SYSTEM_INSTRUCTION}\n\nUser: {user_input}\nChatbot:"
    
    while True:  # Retry loop if server busy
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            print(f"Assistant: {response.text}")
            break
        except ServerError:
            print("Server busy, retrying in 10 seconds...")
            time.sleep(10)
