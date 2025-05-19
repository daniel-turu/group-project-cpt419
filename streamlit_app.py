import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI

# Initialize only once
@st.cache_resource
def load_qa_chain():

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="index_store", embedding_function=embedding)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4", openai_api_key=st.secrets["OPENAI_API_KEY"])
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Set page configuration
st.set_page_config(page_title="FUTMINNA CS Handbook Assistant", layout="wide")

# Main title
st.title("🤖 Department of Computer Science Handbook Assistant")
st.markdown("""
Welcome to the **FUTMINNA Computer Science Handbook Assistant (2024–2028)**.

This tool allows students to easily ask questions and retrieve relevant information from the departmental handbook using AI.

**Project Details:**
- 📘 **Course**: CPT419 - Group 15
- 🏫 **Institution**: Federal University of Technology, Minna
- 🧠 **Focus**: Departmental Handbook (2024–2028 Edition)
""")

# Sidebar for context
st.sidebar.title("📄 About This Assistant")
st.sidebar.markdown("""
This assistant was built using:
- **LangChain** for retrieval-based Q&A
- **OpenAI GPT-4** for natural language understanding
- **HuggingFace Embeddings** for document representation
- **ChromaDB** as the vector store

You can use this tool to ask questions about:
- Course descriptions
- Graduation requirements
- Academic policies
- Staff and office information
- Departmental rules and expectations
""")

# Load QA bot
qa = load_qa_chain()

# User interaction
user_question = st.text_input("💬 Ask a question about the handbook:")

# Output response
if user_question:
    with st.spinner("Searching the departmental handbook..."):
        response = qa.run(user_question)
        st.success(response)
        
