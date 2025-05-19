from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownTextSplitter
from langchain.schema import Document
import os
import json
from langchain_community.document_loaders import TextLoader, JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_docs():
    docs = []
    for filename in os.listdir("data"):
        filepath = os.path.join("data", filename)
        if filename.endswith(".md"):
            loaded = TextLoader(filepath).load()
            # Add metadata and title extraction
            for doc in loaded:
                first_line = doc.page_content.split("\n")[0]
                doc.metadata.update({"source": filename, "title": first_line.strip("# ").strip()})
            docs.extend(loaded)

        elif filename.endswith(".json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Flatten JSON to text for better embeddings
            if filename == "non_academic_staff_records.json":
                content = ""
                for staff in data.get("non_academic_staff", []):
                    line = f"Name: {staff['name']}, Rank: {staff['rank']}, Qualifications: {', '.join(staff['qualifications'])}.\n"
                    content += line
                docs.append(Document(page_content=content, metadata={"source": filename}))

            else:
                # fallback JSON loading for array/dict
                if isinstance(data, list):
                    docs.extend(JSONLoader(filepath, jq_schema=".[]", text_content=False).load())
                elif isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, list):
                            jq_schema = f".{key}[]"
                            docs.extend(JSONLoader(filepath, jq_schema=jq_schema, text_content=False).load())
                            break
                    else:
                        docs.extend(TextLoader(filepath).load())

    return docs

def embed_and_store():
    docs = load_docs()
    # Use Markdown splitter if available for better semantic chunking
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)

    chunks = splitter.split_documents(docs)
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embedding, persist_directory="index_store")
    vectorstore.persist()

if __name__ == "__main__":
    embed_and_store()
