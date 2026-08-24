import os

from groq import Groq
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain import hub
from langchain_core.tools import tool

from langchain.agents import create_react_agent, AgentExecutor

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROK_API_KEY"),
    model="qwen/qwen3.6-27b",
    temperature = 0.0
)

@tool
def multiplicator(multi_expression: str)->str:
    """Multiply two numbers"""
    try:
        result = eval(multi_expression)
        return str(result)
    except Exception as error:
        return f"There is an error {error}"

@tool
def addition(addition_expression: str)->str:
    """Add two numbers"""
    try:
        result = eval(addition_expression)
        return str(result)
    except Exception as error:
        return f"There is an error {error}"

@tool
def dividor(dividor_expression: str)->str:
    """Divide two numbers"""
    try:
        result = eval(dividor_expression)
        return str(result)
    except Exception as error:
        return f"There is an error {error}"

tools = [multiplicator, addition, dividor]
prompts = hub.pull("hwchase17/react")
created_agent = create_react_agent(llm,tools,prompts)

agent_runner = AgentExecutor(
    agent = created_agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

questions = [
    "What is the multiplication of 2 with 5?",
    "What is the value of 3 + 5?",
    "Divide 2 by 0?"
]

for question in questions:
    print(f"Question is {question}")
    result = agent_runner.invoke({"input":question})
    print(f"The answer is {result['output']}")