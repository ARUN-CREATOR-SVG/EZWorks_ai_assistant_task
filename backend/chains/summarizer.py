from llm.llm_api import call_llama_with_context

def generate_summary(text_chunks):
    joined_text = "\n".join(text_chunks[:10])
    return  call_llama_with_context(joined_text, "Give a concise summary of this content in under 150 words")
