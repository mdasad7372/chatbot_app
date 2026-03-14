import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

#1 load pdfs

def load_pdfs(folder):
 docs = []
 for pdf in Path(folder).glob("*.pdf"):
    loader = PyPDFLoader(str(pdf))
    docs.extend(loader.load())
 return docs

documents = load_pdfs("data/pdf")


#2. split documents

splitter = RecursiveCharacterTextSplitter(
  chunk_size = 1000,
  chunk_overlap = 200
)

chunks = splitter.split_documents(documents)

#3 embedding

embeddings = HuggingFaceEmbeddings(
  model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#4 vector store

vectorstore = FAISS.from_documents(chunks,embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

#5 LLM

llm = ChatGroq(
    model_name = "llama-3.1-8b-instant"
)

#6 prompt

prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context.

Context:
{context}

Question:
{question}
""")

#7 RAG chain

rag_chain = (
  {"context":retriever,"question":RunnablePassthrough()}
  | prompt
  | llm
  | StrOutputParser()
)
