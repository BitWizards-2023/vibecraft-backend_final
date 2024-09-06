import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_text_emotion(typed_text: str) -> str:
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Analyze the following text and determine the emotion: {typed_text}",
            max_tokens=50,
            temperature=0.5
        )
        # Extract emotion from the response
        emotion = response.choices[0].text.strip()
        return emotion

    except Exception as e:
        raise Exception(f"Error with OpenAI API: {str(e)}")
