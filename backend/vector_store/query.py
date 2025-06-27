from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

VECTOR_DB_DIR = os.path.join(os.getcwd(), "vector_store", "index")
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def retrieve_relevant_docs(query, filename):
    vectorstore_path = os.path.join(VECTOR_DB_DIR, os.path.splitext(filename)[0])
    if not os.path.exists(vectorstore_path):
        raise FileNotFoundError(f"Vector store not found for file: {filename}")
    db = FAISS.load_local(vectorstore_path, embed_model, allow_dangerous_deserialization=True)
    return db.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(query)

def get_full_context(filename):
    vectorstore_path = os.path.join(VECTOR_DB_DIR, os.path.splitext(filename)[0])
    if not os.path.exists(vectorstore_path):
        raise FileNotFoundError(f"Vector store not found for file: {filename}")
    db = FAISS.load_local(vectorstore_path, embed_model, allow_dangerous_deserialization=True)
    docs = db.similarity_search("core concepts and important content", k=20)
    return "\n".join([doc.page_content for doc in docs])
