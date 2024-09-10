import random
from fastapi import HTTPException
from app.services.text_analysis import analyze_text_with_openai
import logging

logger = logging.getLogger(__name__)

# Define the array of emotions
emotions_array = ["happy", "angry", "sad", "calm", "neutral"]

class TypingController:
    """
    Controller class to handle typing dynamics and emotion detection requests.
    """

    @staticmethod
    async def process_typing_analysis(data):
        """
        Processes the typing data by interacting with both text and typing dynamics analysis.
        
        Args:
            data: The typing data input from the user, typically a Pydantic model.
        
        Returns:
            dict: The result from the typing and emotion analysis.
        """
        try:
            emotion = ""
            if data.text_input:
                logger.info("Analyzing emotion from text input...")
                
                # Step 1: Analyze the text using OpenAI's API
                response = await analyze_text_with_openai(data)
                emotion = response.get('emotion')                
                print(emotion)
                
                # Step 2: If emotion detected from text, return the result
                if emotion != "unidentified":
                    return {"emotion": emotion}
                elif emotion == "unidentified":
                    # Step 3: If emotion not identified, randomly select emotion from emotions_array
                    emotion = random.choice(emotions_array)
                    logger.info(f"Randomly selected emotion: {emotion}")
                    return  {"emotion": emotion}
                
        except HTTPException as e:
            logger.error(f"Error during emotion detection: {e.detail}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during emotion detection: {e}")
            raise HTTPException(status_code=500, detail=f"Error during emotion detection: {e}")
