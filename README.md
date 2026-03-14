#RAG Chatbot with Groq + LangChain + Streamlit

This project implements a Retrieval-Augmented Generation (RAG) system that allows users to ask questions about PDF documents.

The system retrieves relevant document chunks from a vector database and uses a large language model to generate context-aware answers.

#Tech Stack

- Python
- LangChain
- FAISS (Vector Database)
- HuggingFace Sentence Transformers
- Groq (Llama 3 LLM)
- Streamlit

#Architecture

PDF Documents  
↓  
Document Loader  
↓  
Text Chunking  
↓  
Embeddings (Sentence Transformers)  
↓  
FAISS Vector Store  
↓  
Retriever  
↓  
Groq LLM  
↓  
Generated Answer  
↓  
Streamlit UI

#What I Learned

While building this project I learned:

- How Retrieval-Augmented Generation (RAG) works
- Document ingestion and chunking using LangChain
- Creating semantic embeddings for document search
- Using FAISS as a vector database
- Building retrieval pipelines for LLMs
- Prompt design for context-aware responses
- Integrating Groq LLMs with LangChain
- Building a simple AI interface using Streamlit

#Running the Project

```bash
pip install -r requirements.txt
streamlit run app.py



