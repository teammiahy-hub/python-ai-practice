from langchain.memory import ConversationBufferWindowMemory, ConversationTokenBufferMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

llm_model = "openai/gpt-oss-20b"

llm = ChatGroq(
    model= llm_model,
    temperature = 0.0
)

memory = ConversationBufferWindowMemory(k=1)

memory.save_context({"input":"Hi"},{"output":"What's up"})
memory.save_context({"input":"Not much, just hanging"},{"output":"Cool"})

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

print(memory.load_memory_variables({}))

predict_name = conversation.predict(input="Hi, my name is Faniry")
print(predict_name)
predict_sum = conversation.predict(input="What is 1+1")
print(predict_sum)

predict_name = conversation.predict(input="What is my name")
print(predict_name)

memory_token = ConversationTokenBufferMemory(llm=llm, max_token_limit=5)
memory_token.save_context({"input":"AI is what?"},{"output":"Amazing"})
memory_token.save_context({"input":"Baseball is what?"},{"output":"Beautiful"})
memory_token.save_context({"input":"Charm is what?"},{"output":"Cool"})
print(memory_token.load_memory_variables({}))

schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory_summary = ConversationSummaryBufferMemory(llm=llm,max_token_limit = 20)

memory_summary.save_context({"input":"hello"},{"output":"What's up"})
memory_summary.save_context({"input":"not much}, just hanging"},{"output":"Cool"})
memory_summary.save_context({"input":"What is on the schedule today?"},{"output":f"{schedule}"})
print(memory_summary.load_memory_variables({}))
conversation_summary = ConversationChain(
    llm=llm,
    memory=memory_summary,
    verbose = False
)
suggestion = conversation_summary.predict(input="What would be a good demo?")
print(suggestion)
print(memory_summary.load_memory_variables({}))