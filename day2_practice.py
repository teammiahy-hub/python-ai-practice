def calculate_tokens(message):
    words = message.split()
    number_of_tokens = len(words)* 1.3
    return int(number_of_tokens)

message = "This is a sample AI message prompt to calculate tokens"
number_of_tokens = calculate_tokens(message)
print(f"number of tokens {number_of_tokens}")

def create_message(content, role="user"):
    message = {
        "role" : role,
        "content" : content
    }
    return message

message1 = create_message("What is a RAG")
print(message1)

message2 = create_message("What is his role",role = "owner")
print(message2)


def process_documents(documents):
    results = []

    for i, doc in enumerate(documents):
        processed = {
            "id": i,
            "doc": doc,
            "word_counts": len(doc.split()),
            "is_long": len(doc.split()) > 10
        }
        results.append(processed)
    return results

docs = [
    "This is the introduction",
    "This is the development with many words in it elaborating the list",
    "The content of this is for AI practice"
]
processed = process_documents(docs)
for item in processed:
    print(f"Doc: {item['id']}: {item['word_counts']} words: is long = {item['is_long']}")
