import os
import logging
from fastapi import HTTPException
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up logging
logger = logging.getLogger(__name__)

async def analyze_text_with_openai(text: str):
    """
    Analyze the text using OpenAI API and return the detected emotion in a flat structure.
    """
    try:
        # Optimized prompt for detecting emotions
        prompt = f"""
        Analyze the emotional tone of the following text and classify it as one of the following emotions: 'happy', 'sad', 'angry', 'calm', 'neutral'. 
        If the text is mining less and emotion cannot be clearly identified, classify it as 'unidentified'.
        Provide only the emotion label.
        Text: "{text}"
        """

        # Call OpenAI's Chat API for emotion detection using GPT-4
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]        )

      
        # Correctly access the content of the message using dot notation
        emotion = response.choices[0].message.content.strip().lower()

        # Return the extracted emotion in a flat structure
        return {
            "emotion": emotion
        }

    except Exception as e:
        # Log the error if something goes wrong
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")
