from pydantic import BaseModel
from typing import Optional

class SummaryRequest(BaseModel):
    filename: str


class AskRequest(BaseModel):
    question: str
    filename: Optional[str] = None


class EvaluateRequest(BaseModel):
    q: str
    user_answer: str
    filename: Optional[str] = None

class ChallengeRequest(BaseModel):
    filename: str
