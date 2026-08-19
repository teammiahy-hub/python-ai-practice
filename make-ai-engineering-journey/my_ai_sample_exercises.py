import os
from groq import Groq
from dotenv import load_dotenv

# Exercise 1: Write a function that:
# - Takes a list of messages (dictionaries)
# - Counts how many are from "user"
# - Counts how many are from "assistant"
# - Prints both counts
# - Returns total count
def prompt_count(dictionary_list):
    user_count = 0
    assistant_count = 0
    for item in dictionary_list:
        if item["Role"] == "user":
            user_count = user_count + 1
        if item["Role"] == "assistant":
            assistant_count = assistant_count + 1

    print(f"There are {user_count} list of messages from user, and {assistant_count} from assistant")
    return user_count + assistant_count

dictionary_list=[
    {"Role":"user","Content":"One message"},
    {"Role":"system","Content":"One letter"},
    {"Role":"assistant","Content":"One stamp"},
    {"Role":"user","Content":"One cover"}
]

total_count = prompt_count(dictionary_list)
print(f"Total count {total_count}")

# Exercise 2: Write a function that:
# - Takes a long text (string)
# - Splits it into chunks of N words
# - Returns a list of chunks
# - Example: "hello world foo bar" split by 2 = ["hello world", "foo bar"]
def chunck_text(text,split=2):
    words = text.split()
    chuncks = []

    for i in range(0, len(words), split):
        chunck = words[i:i + split]
        chuncks.append(" ".join(chunck))

    return chuncks

result_chunck=chunck_text("hello world foo bar",3)
print(result_chunck)

# Exercise 3: Write a chatbot that:
# - Has a while loop that keeps running
# - Asks user for input
# - Sends it to Groq API
# - Prints response
# - If user types "quit" it stops
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

conversations=[
    {"role":"user","content":"Init"}
]

def live_chat():
    while True:
        question = input("Ask your AI assistant (type 'quit' to exit): ")

        # Exit before adding the question to the conversation
        if question.lower() == "quit":
            print("Exiting...")
            break

        # Add user's message
        conversations.append({
            "role": "user",
            "content": question
        })

        # Send conversation to AI
        answer = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=conversations,
            temperature=0.7,
            max_tokens=512
        )

        # Get AI response
        result = answer.choices[0].message.content

        # Add AI response to conversation
        conversations.append({
            "role": "assistant",
            "content": result
        })

        # Display AI response
        print("AI:", result)

    return conversations


result = live_chat()
