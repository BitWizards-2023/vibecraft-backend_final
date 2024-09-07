import os
import pickle  # Use pickle to load the model
from fastapi import HTTPException
from sklearn.preprocessing import StandardScaler

# Load the trained XGBoost model and scaler
model_path = os.getenv("MODEL_PATH", "app/models/best_xgb_model.pkl")  # Update to .pkl file
scaler_path = os.getenv("SCALER_PATH", "app/models/scaler.pkl")  # Update if scaler is also a .pkl file

try:
    # Load the XGBoost model using pickle
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # Load the scaler using pickle (if scaler is in .pkl format)
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

except FileNotFoundError as e:
    raise HTTPException(status_code=500, detail=f"Model or Scaler file not found: {e}")
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading model or scaler: {e}")

def detect_emotion_with_typing_dynamics(features: list) -> str:
    """
    Detect emotion using typing dynamics with the pre-trained XGBoost model.
    
    Args:
        features (list): A list of features corresponding to the typing dynamics data.
        
    Returns:
        str: Predicted emotion label.
    """
    try:
        # Check if features match the expected size
        if len(features) != scaler.n_features_in_:
            raise ValueError(f"Expected {scaler.n_features_in_} features, but got {len(features)}.")

        # Scale the features
        scaled_features = scaler.transform([features])

        print(scaled_features)

        # Make a prediction using the loaded XGBoost model
        prediction = model.predict(scaled_features)

        # Get the predicted label (if output is probability, handle accordingly)
        predicted_label = int(prediction[0])

        # Mapping predictions to emotions
        emotions = {0: "neutral", 1: "happy", 2: "calm", 3: "sad", 4: "angry"}
        return emotions.get(predicted_label, "Unknown")

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Input feature error: {ve}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error with ML model prediction: {str(e)}")
