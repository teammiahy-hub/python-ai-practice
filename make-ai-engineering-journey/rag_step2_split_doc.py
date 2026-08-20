from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
documents = loader.load() 

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)
chunks = splitter.split_documents(documents)

print(f"Original document: {len(documents)}")
print(f"After splitting: {len(chunks)} chunks")
print(f"\nFirst chunk:")
print(chunks[0].page_content)
print(f"\nSecond chunk:")
print(chunks[1].page_content)