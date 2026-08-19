import os
from dotenv import load_dotenv, find_dotenv

from langchain.chains import RetrievalQA
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.evaluation.qa import QAGenerateChain, QAEvalChain
from langchain_huggingface import HuggingFaceEmbeddings
import langchain
langchain.debug = True

from groq import Groq
from langchain_groq import ChatGroq

_ = load_dotenv(find_dotenv())
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path = file)
data = loader.load()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding = embeddings
).from_loaders([loader])

llm_model = "openai/gpt-oss-20b"
llm=ChatGroq(
    llm=llm_model,
    temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)
print(data[10])
print(data[11])

examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]

example_gen_chain = QAGenerateChain.from_llm(ChatGroq())
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)
print(new_examples[0])
examples += new_examples
# Check what is happening
print(qa.run(examples[0]["query"]))
langchain.debug = False
predictions = qa.apply(examples)
print(predictions)
eval_chain = QAEvalChain.llm_model(llm)
graded_output = eval_chain.evaluate(examples, predictions)
for i,eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()