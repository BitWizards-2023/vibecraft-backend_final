import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

async def analyze_text_with_openai(text: str):
    """
    Analyze the text using OpenAI API and return the detected emotion.
    """
    try:
        # Optimized prompt for detecting emotions
        prompt = f"""
        Analyze the emotional tone of the following text and classify it as one of the following emotions if cannot clearly identified send it under this label "unidentified": 
        'happy', 'sad', 'angry', 'calm', 'neutral'. 
        Provide only the emotion label without any additional explanation.
        Text: "{text}"
        """        
        # Call OpenAI's API for emotion detection
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=10,   # Since we expect a single word (emotion), a lower token limit is optimal
            temperature=0.5
        )
        emotion = response.choices[0].text.strip().lower()
        return emotion
    except Exception as e:
        #logger.error(f"OpenAI API error: {e}")
        raise Exception(status_code=500, detail=f"OpenAI API error: {e}")

