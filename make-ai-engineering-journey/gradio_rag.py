import os
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from groq import Groq

import gradio as gr
# import inspect
# print(gr.__version__)
# print(inspect.signature(gr.Chatbot))

load_dotenv()

api_key=os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    api_key=api_key,
    model="qwen/qwen3.6-27b",
    temperature=0.0
)
# print(dir(llm))
# client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# models = client.models.list()
# for model in models.data:
#     print(model.id)
embedding_model = HuggingFaceEmbeddings(
    model_name ="all-MiniLM-L6-v2"
)

#Global storage
rag_system = {"chain":None}

def load_url(url):
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 50
        )
        chunk = splitter.split_documents(documents)

        vectorestore = Chroma.from_documents(
            documents=chunk,
            embedding=embedding_model
        )

        prompt = PromptTemplate(
            template="""Answer using only the context provided below.

        Context:
        {context}

        Question:
        {question}

        Answer:""",
            input_variables=["context", "question"]
        )

        rag_system["chain"] = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorestore.as_retriever(search_kwargs={"k":3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )

        return f"Loaded {len(chunk)}! Ready to answer questions!"
    except Exception as e:
        return f"Error {e}"

def answer_question(question, history):
    if rag_system["chain"] is None:
        return history + [
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": "Please input URL first."
            }
        ]

    if not question.strip():
        return history

    result = rag_system["chain"].invoke({"query": question})

    print(result)

    answer = result["result"]
    sources = len(result["source_documents"])

    full_answer = (
        f"{answer}\n\n"
        f"Source chunks used: {sources}"
    )

    history.append(
        {
            "role": "user",
            "content": question
        }
    )

    history.append(
        {
            "role": "assistant",
            "content": full_answer
        }
    )

    return history

with gr.Blocks(title="myRAGAPP") as app:
    gr.Markdown("# 📚 Document Q&A System")
    gr.Markdown("Load any webpage and ask questions about it!")
    # func,inputs,outputs
    with gr.Row():
        url_box = gr.Textbox(
            label="Website URL here!",
            placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence",
            scale=4
        )
        load_button = gr.Button("Load",variant="primary",scale=1)

    status_box = gr.Textbox(label="Status",interactive=False)

    chatBot = gr.Chatbot(label="Chat",height=300)

    with gr.Row():
        question_box = gr.Textbox(
            label="Your Question",
            placeholder="Ask anything about the loaded page...",
            scale=4
        )
        ask_button = gr.Button("Ask", variant="primary", scale=1)

    # Link question to function
    load_button.click(
        fn=load_url,
        inputs=[url_box],
        outputs=[status_box]
    )

    ask_button.click(
        fn=answer_question,
        inputs=[question_box, chatBot],
        outputs=[chatBot]
    )
# To generate a public link share to True
print("Starting Gradio...")
app.launch(share=True)
# app.launch(
#     share=True,
#     server_port=7861
# )
print("Gradio stopped")