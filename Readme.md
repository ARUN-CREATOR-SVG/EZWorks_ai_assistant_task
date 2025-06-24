# EZworks AI Assistant 🤖

A GenAI-powered assistant built using **FastAPI**, **LangChain**, **HuggingFace models**, and **vector databases**, designed to:
- Ingest documents (PDFs, text)
- Generate summaries
- Answer questions from uploaded content
- Generate logic-based challenges
- Evaluate user responses with similarity scoring

---

## 🚀 Features

- 📤 **File Upload**: Upload documents to ingest and store embeddings
- 📝 **Summarization**: Automatically summarize uploaded content
- ❓ **Question Answering**: Ask context-based questions from uploaded files
- 🧠 **Challenge Generation**: Auto-generate logic questions from content
- 📊 **Answer Evaluation**: Uses cosine similarity to grade user answers
- 🔗 **LangChain RAG**: Retrieval-Augmented Generation using vector store
- ⚙️ **FastAPI Backend** with modular route handling and OpenAPI docs

---




## 🔧 Tech Stack

- 🐍 **Python 3.10**
- ⚡ **FastAPI**
- 🔗 **LangChain**
- 🤗 **HuggingFace Transformers + Inference API**
- 📁 **FAISS** for vector search
- 📐 **Scikit-learn** for cosine similarity
- 🧪 **Pydantic** for schema validation

---

## 📦 Installation

```bash
git clone https://github.com/ARUN-CREATOR-SVG/EZWorks_ai_assistant_task
cd EZworks_ai_assistant
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
pip install -r backend/requirements.txt



🔐 Environment Variables
Create a .env file in the root directory and add:

HUGGINGFACEHUB_API_TOKEN=your_token_here

▶️ Run the App
uvicorn backend.main:app --reload

Open browser at: http://localhost:8000/docs 🚀


✨ Author
Made with ❤️ by Arun

