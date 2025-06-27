from vector_store.query import retrieve_relevant_docs
from llm.llm_api import call_llama_with_context

def ask_question(user_query, filename):
    context_docs = retrieve_relevant_docs(user_query, filename)
    context_text = "\n".join([doc.page_content for doc in context_docs])
    answer =  call_llama_with_context(context_text, user_query)
    citations = [
        {
            "text": doc.page_content[:100] + "...",
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page"),
            "paragraph": doc.metadata.get("paragraph")
        }
        for doc in context_docs
    ]
    return {"answer": answer, "justified_by": citations}