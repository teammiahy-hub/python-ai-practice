from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
documents = loader.load()

print(f"Number of documents: {len(documents)}")
print(f"Type: {type(documents[0])}")
print(f"Content preview: {documents[0].page_content[:300]}")
print(f"Metadata: {documents[0].metadata}")