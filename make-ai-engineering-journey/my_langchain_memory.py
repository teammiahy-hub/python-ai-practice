from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from groq import Groq

_=load_dotenv(find_dotenv())

memory = ConversationBufferMemory()

llm_model = "openai/gpt-oss-20b"

llm = ChatGroq(
    model=llm_model,
    temperature=0.0
)

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

print(conversation.predict(input="Hi, my name is Faniry"))
# print(conversation.predict(input="What is 1+1? Who said this? and why so ? why not other value?"))
# print(conversation.predict(input="what is my name?"))
# print(memory.buffer)
print(memory.load_memory_variables({}))
memory.save_context({"input": "hi"}, {"output": "what's up?"})
print(memory.buffer)
print(memory.load_memory_variables({}))