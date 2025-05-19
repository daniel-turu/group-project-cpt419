from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI


YOUR_OPENAI_API_KEY = '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'

def create_qa_bot():
    
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="index_store", embedding_function=embedding)
    retriever = vectorstore.as_retriever()
    
    llm = ChatOpenAI(model="gpt-4", openai_api_key=YOUR_OPENAI_API_KEY)  # or "gpt-3.5-turbo"
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

if __name__ == "__main__":
    qa_bot = create_qa_bot()
    while True:
        q = input("Ask a question: ")
        print(qa_bot.run(q))
