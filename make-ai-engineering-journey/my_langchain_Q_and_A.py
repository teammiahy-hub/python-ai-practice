import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from langchain_groq import ChatGroq
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from IPython.display import display, Markdown


# -----------------------------
# 1. LLM
# -----------------------------

llm_model = "openai/gpt-oss-20b"

llm = ChatGroq(
    model=llm_model,
    temperature=0.0
)


# -----------------------------
# 2. Load CSV
# -----------------------------

file = "OutdoorClothingCatalog_1000.csv"

loader = CSVLoader(file_path=file)
documents = loader.load()


# -----------------------------
# 3. Create embeddings
# -----------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# 4. Create vector store
# -----------------------------

vectorstore = FAISS.from_documents(
    documents,
    embeddings
)


# -----------------------------
# 5. Create retriever
# -----------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10}
)


# -----------------------------
# 6. Ask question
# -----------------------------

query = """Please list all your shirts with sun protection
in a table in markdown and summarize each one."""


# Retrieve relevant documents
docs = retriever.invoke(query)


# -----------------------------
# 7. Build context
# -----------------------------

context = "\n\n".join(
    doc.page_content for doc in docs
)


# -----------------------------
# 8. Ask the LLM
# -----------------------------

prompt = f"""
You are an assistant answering questions about an outdoor clothing catalog.

Use ONLY the catalog information provided below.

Catalog information:
{context}

Question:
{query}

If there are matching products, provide:
1. A Markdown table containing the relevant shirts.
2. A short summary of each shirt.

Do not invent products or information that isn't present in the catalog.
"""

response = llm.invoke(prompt)


# -----------------------------
# 9. Display answer
# -----------------------------

display(Markdown(response.content))
