from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

chunk = splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

query = "What is RAG?"
# Embed the query
embedding = embedding_model.embed_query(query)

# Embed also the documents for later similarity search below in vectorstore
vectorstore = Chroma.from_documents(
    documents=chunk,
    embedding=embedding_model,
    persist_directory="./chroma_storage"
)

print(f"Stored {len(chunk)} of chunks in vector db")

query = "How does retrieval augmented generation work?"
similar_docs = vectorstore.similarity_search(query)
print(f"Query: {query}")
print(f"Found {len(similar_docs)} chunks about the query")

for i,doc in enumerate(similar_docs):
    print(f"Result: {i+1}")
    print(f"Doc content: {doc.page_content[:200]}")