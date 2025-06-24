from llm.llama_api import call_llama_with_context
from vector_store.query import get_full_context
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
cached_questions = []

def generate_challenges():
    return {"error": "Filename required for challenge generation."}

def generate_challenges(filename):
    text = get_full_context(filename)
    questions = call_llama_with_context(text, "Generate 3 logic-based questions from the above content")
    global cached_questions
    cached_questions = questions.strip().split("\n")[:3]
    return {"questions": cached_questions}

def evaluate_response(question, user_answer, filename):
    ideal = call_llama_with_context("", f"What is the best answer to: {question}")
    vec_user = embedding_model.embed_query(user_answer)
    vec_ideal = embedding_model.embed_query(ideal)
    score = cosine_similarity([vec_user], [vec_ideal])[0][0]
    justification = f"Cosine similarity with ideal answer: {score:.2f}"
    return {
        "ideal_answer": ideal.strip(),
        "score": round(score, 2),
        "justification": justification
    }
