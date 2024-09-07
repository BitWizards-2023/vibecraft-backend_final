from fastapi import HTTPException
from app.services.text_analysis import analyze_text_with_openai
import logging

logger = logging.getLogger(__name__)

class TypingController:
    """
    Controller class to handle typing dynamics and emotion detection requests.
    """

    @staticmethod
    async def process_typing_analysis(data):
        """
        Processes the typing data by interacting with both text and typing dynamics analysis.
        
        Args:
            data: The typing data input from the user.
        
        Returns:
            dict: The result from the typing and emotion analysis.
        """
        try:
            # Step 1: If text input is provided, analyze the text using OpenAI's API
            if data.text_input:
                logger.info("Analyzing emotion from text input...")
                emotion_from_text = await analyze_text_with_openai(data.text_input)
                
                # If valid emotion detected from text, return the result
                if emotion_from_text:
                    return {"analysis": {"detected_emotion": emotion_from_text}}

            # Handle the case where no text input or emotion is detected
            raise HTTPException(status_code=400, detail="Unable to detect emotion from text input")

        except Exception as e:
            logger.error(f"Error during emotion detection: {e}")
            raise HTTPException(status_code=500, detail=f"Error during emotion detection: {e}")
