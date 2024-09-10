from fastapi import HTTPException
from app.services.text_analysis import analyze_text_with_openai
from app.services.typing_dyanmics_ml_service import detect_emotion_with_typing_dynamics
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
                    # Step 3: If emotion not identified, analyze typing dynamics
                    typing_features = [
                        data.D1D2_mean, data.D1D2_std, data.D1U1_mean, data.D1U1_std, data.D1U2_mean, data.D1U2_std,
                        data.U1D2_mean, data.U1D2_std, data.U1U2_mean, data.U1U2_std,
                        int(data.ageRange), int(data.degree), float(data.delFreq), data.editDistance,
                        bool(data.gender), float(data.leftFreq), data.nbKeystroke, int(data.pcTimeAverage), int(data.status), int(data.typeWith),
                        
                        # Country fields
                        bool(data.country_Angola), bool(data.country_Arabie_Saoudite), bool(data.country_Bahamas), bool(data.country_Barbade),
                        bool(data.country_Belgique), bool(data.country_Belize), bool(data.country_Canada), bool(data.country_France),
                        bool(data.country_Germany), bool(data.country_Italy), bool(data.country_Portugal), bool(data.country_Spain),
                        bool(data.country_Suisse), bool(data.country_Tunisia), bool(data.country_United_Arab_Emirates), bool(data.country_United_States),
                        
                        # TypistType fields
                        bool(data.typistType_One_Finger_Typist), bool(data.typistType_Two_Finger_Typist), bool(data.typistType_Touch_Typist)
                    ]
                    
                    logger.debug(f"Extracted typing features: {typing_features}")
                    
                    # Step 4: Pass the typing dynamics features to the model
                    emotion = detect_emotion_with_typing_dynamics(typing_features)
                    return  {"emotion": emotion}
                
        except HTTPException as e:
            logger.error(f"Error during emotion detection: {e.detail}")
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during emotion detection: {e}")
            raise HTTPException(status_code=500, detail=f"Error during emotion detection: {e}")