from fastapi import HTTPException
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import logging

logger = logging.getLogger(__name__)

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='HuggingFaceH4/zephyr-7b-beta',
    task='text-generation',
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
)

model = ChatHuggingFace(llm=llm)

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables=["context", "question"]
)

parser = StrOutputParser()
chain = prompt | model | parser


def call_llama_with_context(context, question):
    try:
        result = chain.invoke({"context": context, "question": question})
        return result.strip()
    except Exception as e:
        logger.exception("Error in call_llama_with_context")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
        