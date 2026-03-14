import streamlit as st
from main import rag_chain

st.set_page_config(page_title="Pdf Chatbot", page_icon="🤖")

st.title("chat with your Documents")

query = st.text_input("Ask a question about your pdfs")

if query:
    with st.spinner("thinking..."):
        answer = rag_chain.invoke(query)

    st.write("### Answer")
    st.write(answer)

