# Write a chain that:
# 1. Takes a topic as input
# 2. First chain: generates 3 questions about the topic
# 3. Second chain: answers the best question
# 4. Prints both the questions and the answer

import os
from groq import Groq
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROK_API_KEY"),
    model="qwen/qwen3.6-27b",
    temperature=0.6
)

result = {"chain" : None}

def set_topic(topic_input):
    chain_1 = (
        ChatPromptTemplate.from_template("Generate 3 questions about the topic: {topic}")
        | llm
        | StrOutputParser()
    )

    chain_2 = (
        ChatPromptTemplate.from_template("Given the 3 questions {questions}\n"
        "Answer the best question")
        | llm
        | StrOutputParser()
    )

    questions = chain_1.invoke({"topic": topic_input})
    answers = chain_2.invoke({"questions":questions})

    print(f"Questions are: {questions}")
    print(f"The picked question to be answered is {questions}, and answer {answers}")

set_topic("English language")