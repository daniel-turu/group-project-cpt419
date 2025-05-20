from langchain.text_splitter import MarkdownTextSplitter
from langchain.schema import Document
import os

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_docs():
    docs = []
    for filename in os.listdir("data"):
        if filename.endswith(".md"):
            filepath = os.path.join("data", filename)
            loaded = TextLoader(filepath).load()
            for doc in loaded:
                first_line = doc.page_content.split("\n")[0]
                doc.metadata.update({"source": filename, "title": first_line.strip("# ").strip()})
            docs.extend(loaded)
    return docs

def embed_and_store():
    docs = load_docs()
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedding)
    vectorstore.save_local("index_store")

if __name__ == "__main__":
    embed_and_store()
