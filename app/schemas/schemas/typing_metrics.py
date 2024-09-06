from pydantic import BaseModel

class TypingMetrics(BaseModel):
    D1U1: float
    D1D2: float
    typingSpeed: float
    typedText: str
