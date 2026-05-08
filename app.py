import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY"

st.title("Swiggy Annual Report RAG Chatbot")

loader = PyPDFLoader("swiggy_report.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embeddings)

question = st.text_input("Ask a question about Swiggy Annual Report")

if question:
    docs = db.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0.2
    )

    prompt = f"""
Answer the question only from the context below.
If the answer is not found, say: The answer is not available in the Swiggy Annual Report.

Context:
{context}

Question:
{question}
"""


    st.subheader("Answer")
    st.write("Gemini quota exceeded, showing retrieved PDF context instead.")

    st.subheader("Supporting Context")
    for i, doc in enumerate(docs):
        st.write(f"Chunk {i+1}")
        st.write(doc.page_content[:700])