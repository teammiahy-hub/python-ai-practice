import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from groq import Groq
from langchain_groq import ChatGroq
from dotenv import load_dotenv,find_dotenv

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

_ = load_dotenv(find_dotenv())

# Load the document
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
documents = loader.load()

# Split into chunk
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)
chunk = splitter.split_documents(documents)

# Embed
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vectorize
vectorstore = Chroma.from_documents(
    documents=chunk,
    embedding=embedding_model,
    persist_directory="./chroma_storage"
)

# Vector store ready to retrieve from
api_key=os.getenv("CHAT_GROQ_KEY")
llm = ChatGroq(
    api_key=api_key,
    model_name="llama-3.1-70b-versatile",
    temperature=0.0
)
prompt = PromptTemplate(
    template="""Answer the question using only the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k":3}),
    chain_type_kwargs={"prompt":prompt},
    return_source_documents=True
)
print("RAG system ready!")

questions = [
    "What is RAG?",
    "What are the benefits of using RAG?",
    "How does retrieval work in RAG systems?"
]
for question in questions:
    print(f"\nQuestion: {question}")
    result = rag_chain.invoke({"query": question})
    print(f"Answer: {result['result']}")
    print(f"Used {len(result['source_documents'])} chunks")