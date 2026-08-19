import openai
import os
from dotenv import load_dotenv, find_dotenv
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# from langchain.prompts import ChatPromptTemplate

customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """American English \
in a calm and respectful tone
"""

prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

template_string = """ Translate the text \
    that is delimited by triple backticks
    into a style that is {style}.
    text: '''{customer_email}'''"""

customer_email_template = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

customer_style = """American English \
in a calm and respectful tone
"""

service_reply = """Hey there customer, \
the warranty does not cover \
cleaning expenses for your kitchen \
because it's your fault that \
you misused your blender \
by forgetting to put the lid on before \
starting the blender. \
Tough luck! See ya!
"""

service_style_polite = """a polite tone that speaks English pirate"""

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

llm_model = "llama-3.1-8b-instant"
chat = ChatGroq(
    model=llm_model,
    temperature=0.0
)
# print(chat)

prompt_template = ChatPromptTemplate.from_template(template_string)
print(prompt_template.messages[0].prompt)
print(prompt_template.messages[0].prompt.input_variables)

customer_messages = prompt_template.format_messages(style=customer_style,customer_email=customer_email_template)
# print(type(customer_messages))
# print(type(customer_messages[0]))
# print(customer_messages)
customer_response = chat.invoke(customer_messages)
print(customer_response.content)

service_messages = prompt_template.format_messages(style=service_style_polite,customer_email=service_reply)
print(service_messages[0].content)

service_response = chat.invoke(service_messages)
print(service_response.content)

def get_completion(prompt, model="llama-3.1-8b-instant"):
    messages=[
        {"role":"user","content":prompt}
        ]
    response = client.chat.completions.create(
        model = model,
        messages = messages,
        temperature = 0.0,
    )
    return response.choices[0].message.content
# print(prompt)
# answer = get_completion(prompt)
# print(answer)
