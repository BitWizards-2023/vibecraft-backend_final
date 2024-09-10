import os
import logging
from fastapi import HTTPException
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up logging
logger = logging.getLogger(__name__)

async def analyze_text_with_openai(data):
    """
    Analyze the text using OpenAI API and return the detected emotion in a flat structure.
    """
    try:
        # Optimized prompt for detecting emotions
        prompt = f"""
        Analyze the emotional tone of the following text and classify it as one of the following emotions: 
        'happy', 'sad', 'angry', 'calm', or 'neutral'. If the text does not clearly convey an emotion or is meaningless, 
        analyze the user's typing dynamics. I will provide you with sample typing dynamics data. Compare this data with 
        predefined emotion-related typing patterns and determine the most closely matching 
        emotion based on similarities in typing speed, keystroke intervals, backspace usage, and other relevant pattern features.
        Return only the detected emotion.send only emotion.do not send any other texts analyze the typing patterns.if you cannot 
        identified emotion clearly send it as a happy
        sample res send as this way "emotion":"happy".send the replay as a json object do not add any other information 
        get only emotion key and relative values
    
        Happy:{{D1D2_mean: 371.68888888888887, D1D2_std: 163.2681502586738, D1U1_mean: 371.68888888888887, D1U1_std: 163.2681502586738, D1U2_mean: 747.1590909090909, D1U2_std: 211.48081795670348, U1D2_mean: 375.1136363636364, U1D2_std: 163.50757558207286, U1U2_mean: 375.1136363636364, U1U2_std: 163.50757558207286, DelFreq: 2, nbKeystroke: 46, leftFreq: 2}}
        Angry: {{D1D2_mean: 309.5903614457831, D1D2_std: 261.187395219767, D1U1_mean: 309.5903614457831, D1U1_std: 261.187395219767, D1U2_mean: 619.4634146341464, D1U2_std: 443.2363791532437, U1D2_mean: 310.0243902439024, U1D2_std: 262.7454949603346, U1U2_mean: 310.0243902439024, U1U2_std: 262.7454949603346, DelFreq: 21, nbKeystroke: 84, leftFreq: 21}}
        Sad:{{D1D2_mean: 462.5542168674699, D1D2_std: 478.7934132066987, D1U1_mean: 462.5542168674699, D1U1_std: 478.7934132066987, D1U2_mean: 928.1829268292682, D1U2_std: 674.4369286536046, U1D2_mean: 463.5, U1D2_std: 481.6272721392474, U1U2_mean: 463.5, U1U2_std: 481.6272721392474, DelFreq: 21, nbKeystroke: 84, leftFreq: 21}}
        calm:{{D1D2_mean: 369.7674418604651, D1D2_std: 187.68881965438382, D1U1_mean: 369.7674418604651, D1U1_std: 187.68881965438382, D1U2_mean: 743.3095238095239, D1U2_std: 277.9257780329815, U1D2_mean: 373.14285714285717, U1D2_std: 188.6158858128137, U1U2_mean: 373.14285714285717, U1U2_std: 188.6158858128137, DelFreq: 1, nbKeystroke: 44, leftFreq: 1}}
        neutral:{{D1D2_mean: 442.1162790697674, D1D2_std: 163.7990828267476, D1U1_mean: 442.1162790697674, D1U1_std: 163.7990828267476, D1U2_mean: 889.1190476190476, D1U2_std: 245.37250593060884, U1D2_mean: 445.4047619047619, U1D2_std: 164.32893841317176, U1U2_mean: 445.4047619047619, U1U2_std: 164.32893841317176, DelFreq: 1, nbKeystroke: 44, leftFreq: 1}}
        """
    

        print(prompt)
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
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")