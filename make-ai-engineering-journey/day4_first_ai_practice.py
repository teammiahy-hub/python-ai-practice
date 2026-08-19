import os
from groq import Groq
from dotenv import load_dotenv

# Load your API key from .env file
load_dotenv()

# Create client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Function to ask AI a question
def ask_ai(question):
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.2,
        max_tokens=512
    )
    return response.choices[0].message.content

answer = ask_ai("What is RAG in AI? Explain in 1 sentences.")
print(answer)