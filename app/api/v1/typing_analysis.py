# app/api/v1/typing_analysis.py
from fastapi import APIRouter
from app.schemas.typing_metrics import TypingData, AnalysisResult
from app.controllers.typing_controller import TypingController

router = APIRouter()

# Health check endpoint
@router.get("/health", response_model=dict)
async def health_check():
    return {"status": "healthy"}

# Typing analysis endpoint
@router.post("/typing_analysis", response_model=AnalysisResult)
async def typing_analysis(data: TypingData):
    # Call the controller, which handles the business logic
    result = await TypingController.process_typing_analysis(data)
    return result
