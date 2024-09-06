import joblib
from fastapi import HTTPException

# Load the trained XGBoost model and scaler
model = joblib.load('path/to/your/xgboost_model.joblib')
scaler = joblib.load('path/to/your/scaler.joblib')

def detect_emotion_with_typing_dynamics(features: list) -> str:
    """
    Detect emotion using typing dynamics with the pre-trained XGBoost model.
    """
    try:
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]
        emotions = {0: "Neutral", 1: "Happy", 2: "Calm", 3: "Sad", 4: "Angry"}
        return emotions[prediction]
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error with ML model prediction")
