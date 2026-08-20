from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)
chunks = splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

embedding = embedding_model.embed_query("What is RAG?")
print(f"Embedding type : {type(embedding)}")
print(f"Embedding len: {len(embedding)}")
print(f"First 5 numbers: {embedding[:5]}")
print(f"These numbers represent the meaning of the text")
