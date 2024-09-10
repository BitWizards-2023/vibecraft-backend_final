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
        Analyze the emotional tone of the following text and classify it as one of the following emotions: 'happy', 'sad', 'angry', 'calm', 'neutral'. 
        If the text is meaning less analyze the touch dynamics according this criteria,with i will give you a predefined typing dynamics
        analysis features & current captured typing dynamics in current typing analysis data set compare that data with predefined data can predict emotions with most familiar  
        emotion : Text"${data.text_input}",typingAnalysisData:"${data}.analyze the pattern of the typingAnalysisData data"
        1. Happy:
            typingAnalysisData: {{D1D2_mean: 371.68888888888887, D1D2_std: 163.2681502586738, D1U1_mean: 371.68888888888887, D1U1_std: 163.2681502586738, D1U2_mean: 747.1590909090909, D1U2_std: 211.48081795670348, U1D2_mean: 375.1136363636364, U1D2_std: 163.50757558207286, U1U2_mean: 375.1136363636364, U1U2_std: 163.50757558207286, DelFreq: 2, nbKeystroke: 46, leftFreq: 2, text_input: The quick brown fox jump over the lazy dog}}
            Typing Speed: Generally, a happy person tends to type faster with fewer errors. Their mood leads to a more confident and fluid typing style.
            Keystroke Intervals (D1D2, U1D2): Shorter intervals between key presses (D1D2) and key releases (U1D2) due to higher energy and smoother flow.
            Backspace Usage (delFreq): Fewer mistakes might lead to lower backspace frequency.
            Keystroke Count (nbKeystroke): Higher overall keystroke count if they're typing quickly, but with fewer corrections.
        2. Angry:
            typingAnalysisData: {{D1D2_mean: 309.5903614457831, D1D2_std: 261.187395219767, D1U1_mean: 309.5903614457831, D1U1_std: 261.187395219767, D1U2_mean: 619.4634146341464, D1U2_std: 443.2363791532437, U1D2_mean: 310.0243902439024, U1D2_std: 262.7454949603346, U1U2_mean: 310.0243902439024, U1U2_std: 262.7454949603346, DelFreq: 21, nbKeystroke: 84, leftFreq: 21, text_input: the quick brown fox jump over the lazy dog}}
            Typing Speed: An angry person might type erratically—either much faster or harder, with more force applied to keys.
            Keystroke Intervals (D1D2, U1D2): The intervals may vary significantly. You could see shorter or inconsistent key press intervals as they may hit keys with more force or aggression.
            Backspace Usage (delFreq): High frequency of backspace use as an angry person may make more errors due to aggressive typing.
            Keystroke Count (nbKeystroke): High keystroke count due to aggressive, quick typing, but possibly also high error rates.
        3. Sad:
            typingAnalysisData:{{D1D2_mean: 462.5542168674699, D1D2_std: 478.7934132066987, D1U1_mean: 462.5542168674699, D1U1_std: 478.7934132066987, D1U2_mean: 928.1829268292682, D1U2_std: 674.4369286536046, U1D2_mean: 463.5, U1D2_std: 481.6272721392474, U1U2_mean: 463.5, U1U2_std: 481.6272721392474, DelFreq: 21, nbKeystroke: 84, leftFreq: 21, text_input: the quick brown fox jump over the lazy dog}}
            Typing Speed: A sad person is likely to type slower, with more hesitation and longer pauses between key presses.
            Keystroke Intervals (D1D2, U1D2): Longer intervals between key presses and releases due to slower typing and hesitation.
            Backspace Usage (delFreq): Increased backspace usage as their slower typing might lead to more self-correction or indecision.
            Keystroke Count (nbKeystroke): Lower overall keystroke count compared to faster emotions like happy or angry.
        4. Calm:
            typingAnalysisData: {{D1D2_mean: 369.7674418604651, D1D2_std: 187.68881965438382, D1U1_mean: 369.7674418604651, D1U1_std: 187.68881965438382, D1U2_mean: 743.3095238095239, D1U2_std: 277.9257780329815, U1D2_mean: 373.14285714285717, U1D2_std: 188.6158858128137, U1U2_mean: 373.14285714285717, U1U2_std: 188.6158858128137, DelFreq: 1, nbKeystroke: 44, leftFreq: 1, text_input: the quick brown fox jump over the lazy dog}}
            Typing Speed: A calm person tends to type consistently and at a steady pace without rushing or making many errors.
            Keystroke Intervals (D1D2, U1D2): Balanced, even intervals between key presses and releases. Calmness leads to a rhythmic and smooth typing pattern.
            Backspace Usage (delFreq): Minimal backspace use due to the composed, deliberate typing style.
            Keystroke Count (nbKeystroke): Moderate keystroke count with fewer corrections compared to other emotional states.
        5. Neutral:
            typingAnalysisData: {{D1D2_mean: 442.1162790697674, D1D2_std: 163.7990828267476, D1U1_mean: 442.1162790697674, D1U1_std: 163.7990828267476, D1U2_mean: 889.1190476190476, D1U2_std: 245.37250593060884, U1D2_mean: 445.4047619047619, U1D2_std: 164.32893841317176, U1U2_mean: 445.4047619047619, U1U2_std: 164.32893841317176, DelFreq: 1, nbKeystroke: 44, leftFreq: 1, text_input: the quick brown fox jump over the lazy dog}}
            Typing Speed: A neutral person is likely to exhibit average typing speed, similar to calm but without a distinctive emotional influence.
            Keystroke Intervals (D1D2, U1D2): Typing intervals are steady, but not as smooth or balanced as when calm. There’s no emotional influence to speed up or slow down typing.
            Backspace Usage (delFreq): Normal backspace frequency, reflecting average levels of self-correction.
            Keystroke Count (nbKeystroke): Keystroke count will reflect an average level of typing activity, neither excessively fast nor slow. 
        Provide only the emotion label.
   
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