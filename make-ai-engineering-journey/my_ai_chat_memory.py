import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

conversations=[
    {
        "role":"system",
        "content":"Introduction"
    }
]
def chat_ctxt(user_message):
    # Add user message to history
    conversations.append({
        "role": "user",
        "content": user_message
    })
    
    # Get AI response
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=conversations,
        temperature=0.7,
        max_tokens=512
    )
    
    # Extract response text
    ai_message = response.choices[0].message.content
    
    # Add AI response to history
    conversations.append({
        "role": "assistant",
        "content": ai_message
    })
    
    return ai_message

# Have a real conversation
print("Me: My name is Alex and I'm learning Python")
print(f"AI: {chat_ctxt('My name is Alex and I am learning Python')}")

print("\nMe: What is my name?")
print(f"AI: {chat_ctxt('What is my name?')}")
# Should remember your name!

print("\nMe: What am I learning?")
print(f"AI: {chat_ctxt('What am I learning?')}")
# Should remember Python!