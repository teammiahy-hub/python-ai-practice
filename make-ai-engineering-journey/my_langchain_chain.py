import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from groq import Groq
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

_=load_dotenv(find_dotenv())
df = pd.read_csv('Data.csv')
print(df.head())
llm_model = "openai/gpt-oss-20b"
llm = ChatGroq(
    model=llm_model,
    temperature=0.9
)
prompt = ChatPromptTemplate.from_template("What is the best name to describe \
    a company that makes {product}?")
chain = LLMChain(llm=llm,prompt=prompt)
product = "Quees Size sheet set"
print(chain.run(product))

prompt_two = ChatPromptTemplate.from_template("Write a 20 words description for the following \
    company:{company_name}")
chain_two = LLMChain(llm=llm,prompt=prompt_two)
overall_simple_chain = SimpleSequentialChain(chains=[chain,chain_two],verbose=True)
print(overall_simple_chain.run(product))
prompt_sequential_one = ChatPromptTemplate.from_template(
    "Translate the following review to English:"
    "\n\n{Review}"
)

chain_sequential_one = LLMChain(
    llm=llm,
    prompt=prompt_sequential_one,
    output_key="English_Review"
)


prompt_sequential_two = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:"
    "\n\n{English_Review}"
)

chain_sequential_two = LLMChain(
    llm=llm,
    prompt=prompt_sequential_two,
    output_key="summary"
)


prompt_sequential_three = ChatPromptTemplate.from_template(
    "What language is the following review?"
    "\n\n{Review}"
)

chain_sequential_three = LLMChain(
    llm=llm,
    prompt=prompt_sequential_three,
    output_key="language"
)


prompt_sequential_four = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}"
    "\n\nLanguage: {language}"
)

chain_sequential_four = LLMChain(
    llm=llm,
    prompt=prompt_sequential_four,
    output_key="followup_message"
)


overall_chain = SequentialChain(
    chains=[
        chain_sequential_one,
        chain_sequential_two,
        chain_sequential_three,
        chain_sequential_four
    ],
    input_variables=[
        "Review"
    ],
    output_variables=[
        "English_Review",
        "summary",
        "language",
        "followup_message"
    ],
    verbose=True
)

data = {
    "Review": [
        "The queen size sheets are extremely comfortable and soft. The fabric feels premium and I sleep better every night.",
        "I love the quality of this bedding set. The colors are beautiful and the material feels very durable.",
        "The product looks elegant but the sheets are not as breathable as expected. It gets warm during the night.",
        "Amazing comfort and excellent craftsmanship. The packaging was also very impressive.",
        "The sheets are good quality for the price, but I wish there were more color options available.",
        "Luxurious queen size bedding crafted for comfort, elegance, and eco responsibility, blending royal heritage with modern minimalism for a restful sleep experience."
    ]
}

df1 = pd.DataFrame(data)

review = df1.Review[5]

result = overall_chain.invoke({
    "Review": review
})

print(result)
