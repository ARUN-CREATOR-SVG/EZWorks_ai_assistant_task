from llm.llm_api import call_llama_with_context
from vector_store.query import get_full_context
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import json

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


challenge_prompt = PromptTemplate(
    input_variables=["context"],
    template="""
You are a helpful AI. Based on the content below, generate exactly 3 clear, logic-based, and self-contained challenge questions.

❌ Avoid incomplete lines, special tags,incomplete text.
✅ Output should be in JSON format exactly like this:

{{
  "questions": [
    "What is ...?",
    "How does ... work?",
    "Why is ... important?"
  ]
}}

Content:
{context}
"""
)


evaluation_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an expert AI assistant.

Q: {question}

Give a clear and concise expert-level answer in **under 100 words**.  
Do NOT add anything unrelated or external examples unless explicitly asked.

Respond only with the answer.
"""
)



def generate_challenges(filename: str):
    context = get_full_context(filename)
    prompt = challenge_prompt.format(context=context[:2000])
    response = call_llama_with_context("", prompt)

    try:
        result = json.loads(response)
        questions = [q.strip() for q in result.get("questions", []) if len(q.strip()) > 10]
    except Exception:
       
        lines = response.strip().split("\n")
        questions = [line.strip("-•123. ") for line in lines if len(line.strip()) > 10]

    
    return {"questions": questions[:3] if questions else ["[ERROR] No valid questions found"]}



def evaluate_response(question: str, user_answer: str, filename: str):
    prompt = evaluation_prompt.format(question=question)
    ideal = call_llama_with_context(get_full_context(filename), prompt)

    vec_user = embedding_model.embed_query(user_answer)
    vec_ideal = embedding_model.embed_query(ideal)
    score = cosine_similarity([vec_user], [vec_ideal])[0][0]

    return {
        "question": question.strip(),
        "user_answer": user_answer.strip(),
        "ideal_answer": ideal.strip(),
        "score": round(score, 2),
        "justification": f"Cosine similarity with ideal answer: {score:.2f}"
    }
