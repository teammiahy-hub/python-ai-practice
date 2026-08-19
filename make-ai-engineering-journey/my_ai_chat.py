# Close the example
# Open new file: my_ai_chat.py
# Write a program that:

# 1. Creates a Groq client
# 2. Has a function called chat(message)
#    that sends message to AI and returns response
# 3. Asks AI these 3 questions and prints answers:
#    - "What is Python?"
#    - "What is machine learning?"
#    - "What is a large language model?"
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def chat(message):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": message
            }
        ],
        temperature=0.4,
        max_tokens=256
    )
    return response.choices[0].message.content

answer = chat("What is Python?")
print(answer)
answer = chat("What is machine learning?")
print(answer)
answer = chat("What is a large language model?")
print(answer)