# app/controllers/typing_controller.py
from app.services.typing_service import analyze_typing
from app.models.typing import TypingData

def process_typing_analysis(data: TypingData):
    """
    Controller function that handles the input, interacts with the business logic,
    and returns the response.
    """
    # Call the business logic from the services layer
    result = analyze_typing(data.dict())
    return result
