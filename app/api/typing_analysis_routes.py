from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class TypingMetrics(BaseModel):
    D1U1: float
    D1D2: float
    typingSpeed: float
    typedText: str

@router.post("/typing_analysis")
async def analyze_typing(metrics: TypingMetrics):
    # Perform your typing analysis here
    # For example, return a simple response with the calculated data
    return {"status": "success", "data": metrics}