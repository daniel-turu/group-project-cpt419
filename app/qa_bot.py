from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

def create_qa_bot():
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load the FAISS index from the saved directory
    vectorstore = FAISS.load_local("index_store", embedding)
    retriever = vectorstore.as_retriever()
    
    llm = ChatOpenAI(model="gpt-4", openai_api_key="YOUR_OPENAI_API_KEY")  # replace with your key or st.secrets
    
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

if __name__ == "__main__":
    qa_bot = create_qa_bot()
    while True:
        q = input("Ask a question: ")
        print(qa_bot.run(q))
