from pydantic import BaseModel
from typing import Optional

class AskRequest(BaseModel):
    question: str
    filename: Optional[str] = None
