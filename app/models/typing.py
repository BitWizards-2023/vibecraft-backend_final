# app/models/typing.py
from pydantic import BaseModel

class TypingData(BaseModel):
    typed_text: str
    typing_metrics: dict

class AnalysisResult(BaseModel):
    analysis: str
