from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from groq import Groq
import os

load_dotenv()

llm_model = "llama-3.1-8b-instant"
chat = ChatGroq(
    model=llm_model,
    temperature=0.7
)
text = {
    "gift": False,
    "delivery_days": 5,
    "price_value":"pretty affordable!"
}
# print(data)
customer_review = """\
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

review_template = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product \
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

Format the output as JSON with the following keys:
gift
delivery_days
price_value

text: {text}
"""

prompt_template = ChatPromptTemplate.from_template(review_template)
print(prompt_template)
messages = prompt_template.format_messages(text = customer_review)
response = chat.invoke(messages)
print(response.content)
print(type(response.content))
gift_schema = ResponseSchema(name="gift",description="Was the item purchased\
                             as a gift for someone else? \
                             Answer True if yes,\
                             False if not or unknown.")
deliveray_day_schema = ResponseSchema(name="delivery_days",description="How many days\
                                      did it take for the product\
                                      to arrive? If this \
                                      information is not found,\
                                      output -1.")
price_value_schema = ResponseSchema(name="price_value",description="Extract any\
                                    sentences about the value or \
                                    price, and output them as a \
                                    comma separated Python list.")
response_schema = [gift_schema, deliveray_day_schema, price_value_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schema)
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

review_template_2 = """\
For the following text, extract the following information:

gift: Was the item purchased as a gift for someone else? \
Answer True if yes, False if not or unknown.

delivery_days: How many days did it take for the product\
to arrive? If this information is not found, output -1.

price_value: Extract any sentences about the value or price,\
and output them as a comma separated Python list.

text: {text}

{format_instructions}
"""
prompt = ChatPromptTemplate.from_template(template=review_template_2)
messages = prompt.format_messages(text=customer_review,format_instructions=format_instructions)
print(messages[0].content)
responses = chat.invoke(messages)
print(responses.content)
output_dict = output_parser.parse(responses.content)
print(output_dict)
print(type(output_dict))
print(output_dict.get("gift"))
print(output_dict.get("delivery_days"))
print(output_dict.get("price_value"))