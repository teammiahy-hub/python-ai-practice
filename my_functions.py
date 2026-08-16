import time
import json
# 1. A function called 'count_characters' that:
#    - Takes a text string as input
#    - Returns the number of characters
#    - Prints a message like "Text has 45 characters"
def count_characters(text):
    count = len(text.split())
    return count

text = "This is a test string"
count = count_characters(text)
print(f"Text has {count} characters")

# 2. A function called 'make_chunks' that:
#    - Takes a list of sentences
#    - Returns only sentences longer than 5 words
def make_chunks(sentences):
    results = []

    for i, doc in enumerate(sentences):
        processed = {
            "id" : i,
            "word_counts": len(doc.split()),
            "is_5_long": len(doc.split()) >= 5
        }
        if processed['is_5_long']:
            results.append(processed)
        # print(f"Is it: {processed['is_5_long']}")
    return results

sentence_list = [
    "This is the 1st sentence",
    "This is the second",
    "This is the 3rd one"
]

processed_sentences = make_chunks(sentence_list)
for item in processed_sentences:
    print(f">=5 words sentences are: {item['id']}: {item['word_counts']} words")

# 3. A function called 'build_prompt' that:
#    - Takes: question (string), context (string)
#    - Returns a formatted string combining them
def build_prompt(question, context="default context"):
    formatted_string = {
        "question": question,
        "context": context
    }
    return formatted_string

text = "This is the introduction of an AI project"
ctxt = "For achieving a special project"

result_string_default = build_prompt(text)
print(f"This is the default formatted string combined: {result_string_default}")

result_string = build_prompt(text, ctxt)
print(f"This is the formatted string combined: {result_string}")


# Dictionaries are how AI messages are structured
llm_config = {
    "api_keys": "your_api_keys",
    "model": "llm_ollama_1.1.0",
    "temperature": 0.7,
    "max_tokens": 512
}

print(f"LLM max tokens: {llm_config["max_tokens"]}")

llm_config["max_tokens"] = 1024
llm_config["extensions"] = "RAG support"

if "temperature" in llm_config:
    print("Temperature is set in llm_config")

for item, key in llm_config.items():
    print(f"{item}: {key}")

api_responses = {
    "id": "response-123",
    "model": "llma-ollama-123",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "This is a demo",
                "content": "This is a RAG system ollama-based"
            }
        }
    ],
    "extensions" : {
        "prompt_tokens": 25,
        "max_tokens": 512,
        "configured_tokens": 64
    }
}

answer = api_responses["choices"][0]["message"]["content"]
extension = api_responses["extensions"]["configured_tokens"]

print(f"{answer}, {extension}")

# Basic try/except
def safe_divide(a,b):
    try:
        result = a/b
        return result
    except ZeroDivisionError:
        print("Divide by zero return none")
        return None

print(safe_divide(2,9))
print(safe_divide(2,0))

# Error handling for API calls
def call_api_safely(prompt,max_retries=3):
    for attempt in range(max_retries):
        try:
            if attempt < 5:
                raise Exception("Temporarily unavailable API")
            return f"Response to {prompt}"
        except Exception as e:
            waiting_time = 2 ** attempt
            print(f"attempt no.{attempt+1} failed with error: {e}")
            print(f"Waiting time {waiting_time} seconds")
            time.sleep(waiting_time)
    return f"Failed to execute API for some reasons"

response = call_api_safely("What is RAG in AI?",2)
print(response)

def save_conversations(messages, filename="conversation_file.json"):
    with open(filename,"w") as file:
        json.dump(messages, file, indent=2)
    print(f"Saved {messages} in {filename}")

def load_conversations(filename):
    try:
        with open(filename, "r") as file:
            messages = json.load(file)
            print(f"Loaded the len of content {len(messages)}")
            return messages
    except FileNotFoundError:
        print("File does not exist")
        return []

conversations = [
    {"role": "owner","content":"I am the owner of this AI"},
    {"role": "RAG","content": "what is RAG?"}
]
save_conversations(conversations)
content = load_conversations("conversation_file.json")
print(content)
    