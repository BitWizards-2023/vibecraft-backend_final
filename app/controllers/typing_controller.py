# app/controllers/typing_controller.py
from app.services.typing_service import analyze_typing
from app.services.text_analysis import analyze_text_with_openai
from app.models.typing import TypingData
import logging

logger = logging.getLogger(__name__)

class TypingController:
    """
    Controller class to handle typing dynamics and emotion detection requests.
    """

    @staticmethod
    async def process_typing_analysis(data: TypingData):
        """
        Processes the typing data by interacting with both text and typing dynamics analysis.
        
        Args:
            data (TypingData): The typing data input from the user.
        
        Returns:
            dict: The result from the typing and emotion analysis.
        """
        try:
            # Step 1: If text input is provided, analyze the text using OpenAI's API
            if data.text_input:
                logger.info("Analyzing emotion from text input...")
                emotion_from_text = await analyze_text_with_openai(data.text_input)
                
                # If valid emotion detected from text, return the result
                if emotion_from_text and emotion_from_text.lower() in ['happy', 'sad', 'calm', 'angry', 'neutral']:
                    return {"detected_emotion": emotion_from_text}

            # Step 2: If text analysis doesn't work or no text provided, analyze typing dynamics
            logger.info("Analyzing emotion from typing dynamics...")
            result = analyze_typing(data.dict())
            
            # Return the detected emotion or analysis result
            return result

        except Exception as e:
            logger.error(f"Error during emotion detection: {e}")
            raise ValueError(f"Failed to analyze typing dynamics: {str(e)}")
