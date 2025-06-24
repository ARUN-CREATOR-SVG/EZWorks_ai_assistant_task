from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from utils.pdf_parser import parse_document
import os

VECTOR_DB_DIR = "vector_store/index"
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

def ingest_document(file_path):
    raw_paragraphs = parse_document(file_path)
    texts = [p["content"] for p in raw_paragraphs]
    metadata = [
        {
            "source": os.path.basename(file_path),
            "page": p["page"],
            "paragraph": p["paragraph"]
        }
        for p in raw_paragraphs
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.create_documents(texts, metadatas=metadata)

    embed_model = HuggingFaceEmbeddings(
         model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore_path = os.path.join(VECTOR_DB_DIR, os.path.splitext(os.path.basename(file_path))[0])
    vectorstore = FAISS.from_documents(documents, embed_model)
    vectorstore.save_local(vectorstore_path)
    return texts