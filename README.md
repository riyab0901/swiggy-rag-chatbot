# Swiggy Annual Report RAG Chatbot

## Objective
This project is a Retrieval-Augmented Generation (RAG) chatbot built using the Swiggy Annual Report PDF. It answers user questions only from the uploaded document.

## Features
- PDF document loading
- Text chunking
- Embedding generation
- FAISS vector database
- Semantic similarity search
- Streamlit user interface
- Context-based question answering

## Technologies Used
- Python
- Streamlit
- LangChain
- FAISS
- Google Gemini API

## Project Workflow
1. Load the Swiggy Annual Report PDF
2. Split text into chunks
3. Generate embeddings
4. Store embeddings in FAISS vector database
5. Retrieve relevant chunks based on user query
6. Generate answers from retrieved context

## How to Run
Install dependencies:

```bash
pip install streamlit langchain langchain-community langchain-google-genai faiss-cpu pypdf