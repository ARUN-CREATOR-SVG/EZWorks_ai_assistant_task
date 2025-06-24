from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from chains.qa_chain import ask_question
from schemas.schemas import AskRequest,EvaluateRequest,ChallengeRequest
from chains.summarizer import generate_summary
from chains.challenge_chain import generate_challenges, evaluate_response
from vector_store.ingest import ingest_document
import shutil, os, tempfile

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    texts = ingest_document(file_path)
    summary =  generate_summary(texts)
    return {"summary": summary}

@app.post("/ask")
def ask(payload: AskRequest):
    return ask_question(payload.question, payload.filename)

@app.post("/challenge")
async def challenge(payload:ChallengeRequest):
    return generate_challenges(payload.filename)

@app.post("/evaluate")
def evaluate(payload: EvaluateRequest):
    return evaluate_response(payload.q, payload.user_answer, payload.filename)