from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import joblib
import soundfile as sf
import os
import librosa
from pydub import AudioSegment
import shutil
 
app = APIRouter()
 
# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Model paths and loading
model_path = os.path.join(
    current_dir, '../../models/Emotion_Voice_Detection_MLPClassifier_Model.pkl')
 
# Ensure the model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"The model file at {model_path} does not exist.")
 
# Load the trained model
model = joblib.load(model_path)
 
# Emotions in the RAVDESS dataset
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
 
# These are the emotions the user wants to observe more
observed_emotions = list(emotions.values())
 
# Function to extract features
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    try:
        # Check if the file is in .m4a format and handle it directly
        if file_name.endswith('.m4a'):
            audio = AudioSegment.from_file(file_name, format="m4a")
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            sample_rate = audio.frame_rate
 
            # Normalize the audio data
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                samples = samples.mean(axis=1)  # Convert to mono by averaging channels
            X = samples / (2 ** 15)  # Convert from 16-bit PCM to float32
        else:
            with sf.SoundFile(file_name) as sound_file:
                X = sound_file.read(dtype="float32")
                sample_rate = sound_file.samplerate
 
        feature_list = []
        if chroma:
            stft = np.abs(librosa.stft(X, n_fft=min(2048, len(X))))
 
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            feature_list.append(mfccs)
 
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            feature_list.append(chroma)
 
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            feature_list.append(mel)
 
        # Concatenate all features
        result = np.concatenate(feature_list, axis=0)
 
        # Adjust feature length to a fixed size (180)
        expected_shape = 180
        if result.shape[0] < expected_shape:
            padding = np.zeros((expected_shape - result.shape[0],))
            result = np.concatenate((result, padding))
        elif result.shape[0] > expected_shape:
            result = result[:expected_shape]
 
        return result
    except Exception as e:
        raise Exception(f"Error extracting features from audio: {e}")
 
def extract_features(file_path):
    # Extract features from audio file
    features = extract_feature(file_path)
    return np.expand_dims(features, axis=0)  # Add batch dimension
 
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the file temporarily
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
 
        # Extract features
        features = extract_features(file_path)
        # Ensure the features are in the correct shape
        if features.shape[1] != model.n_features_in_:
            raise ValueError(f"Feature shape mismatch: expected {model.n_features_in_}, got {features.shape[1]}")
 
        # Make the prediction
        prediction = model.predict(features)
 
        # Determine the predicted emotion
        predicted_emotion = emotions[str(prediction[0]).zfill(2)]
 
        # Clean up the uploaded file
        os.remove(file_path)
 
        return JSONResponse(content={"emotion": predicted_emotion})
 
    except Exception as e:
        # Clean up the uploaded file if an error occurs
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")