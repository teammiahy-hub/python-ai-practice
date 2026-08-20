import requests
import json

def ask_local_ai(question,model="llama3.2"):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model" : model,
            "messages": [
                {"role":"user","content": question}
            ],
            "stream": False
        }
    )
    print(f"Response http: {response.status_code}")
    print(f"Raw response: {response.text}")

    response.raise_for_status()
    data = response.json()
    if "message" not in data:
        raise RuntimeError(f"Unexpected ollama response {data}")
    return data["message"]["content"]

question = "What is your name?"
answer = ask_local_ai(question)
print(f"{question}")
print(f"{answer}")
