from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import joblib
import soundfile as sf
import os
import librosa
from pydub import AudioSegment
import shutil
import speech_recognition as sr
from app.controllers.typing_controller import TypingController

app = APIRouter()

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Model paths and loading
# Corrected model path
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


def get_raw_text_from_audio(file):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(file) as source:
            audio = r.listen(source)
            try:
                text = r.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                return "Error: Speech not recognized"
            except sr.RequestError as e:
                return f"Error: Could not request results; {e}"
    except Exception as e:
        return f"Error: Unable to process file; {e}"

# Function to extract features


def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    try:
        print(get_raw_text_from_audio(file_name))
        transcribed_text = TypingController.transcribe_audio(file_name)
        print("Transcribed Text:", transcribed_text)

        # Check if the file is in .m4a format and handle it directly
        if file_name.endswith('.m4a'):
            audio = AudioSegment.from_file(file_name, format="m4a")
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            sample_rate = audio.frame_rate

            # Normalize the audio data
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                # Convert to mono by averaging channels
                samples = samples.mean(axis=1)
            X = samples / (2 ** 15)  # Convert from 16-bit PCM to float32
        else:
            with sf.SoundFile(file_name) as sound_file:
                X = sound_file.read(dtype="float32")
                sample_rate = sound_file.samplerate

        feature_list = []
        if chroma:
            stft = np.abs(librosa.stft(X, n_fft=min(2048, len(X))))

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(
                y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            feature_list.append(mfccs)

        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate).T, axis=0)
            feature_list.append(chroma)

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(
                y=X, sr=sample_rate).T, axis=0)
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

        # Check if the feature length exceeds 80
        if result.shape[0] > 180:
            return None  # Return None to indicate "Unable to identify"

        return result
    except Exception as e:
        raise Exception(f"Error extracting features from audio: {e}")

# Function to check if the audio has significant features


def is_silent_or_low_energy(features, threshold=0.01):
    variance = np.var(features)
    if variance < threshold:
        return True
    return False


def is_silent_or_low_energy_signal(signal, threshold=1e-4):
    energy = np.sum(signal**2)
    if energy < threshold:
        return True
    return False

# Function to extract features with validation for the expected shape


def extract_features(file_path):
    # Extract features from audio file
    features = extract_feature(file_path)

    # Check for silent or low-energy audio or if shape exceeds 80
    if features is None or is_silent_or_low_energy(features) or is_silent_or_low_energy_signal(features):
        return None  # Indicate no valid features extracted

    return np.expand_dims(features, axis=0)  # Add batch dimension


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Define the upload directory
        upload_path = os.path.join(current_dir, '../../uploads')

        # Ensure the upload directory exists or create it
        try:
            os.makedirs(upload_path, exist_ok=True)
        except OSError as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to create directory: {str(e)}")

        # Define the full file path
        file_path = os.path.join(upload_path, file.filename)

        # Save the file temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract features
        features = extract_features(file_path)

        # Handle the case where no valid features were extracted
        if features is None:
            os.remove(file_path)
            return JSONResponse(content={"emotion": "Unable to identify. Please provide a clearer audio sample."})

        # Ensure the features are in the correct shape
        if features.shape[1] != model.n_features_in_:
            raise ValueError(
                f"Feature shape mismatch: expected {model.n_features_in_}, got {features.shape[1]}")

        # Make the prediction
        prediction = model.predict(features)
        print(prediction)

        # Ensure the prediction is properly handled
        prediction_value = prediction[0]

        # If the prediction is numeric (as expected)
        if isinstance(prediction_value, (int, np.integer)):
            predicted_emotion = emotions.get(
                str(prediction_value).zfill(2), "Unknown")
        # If the prediction returns a string (e.g., 'surprised')
        elif isinstance(prediction_value, str):
            predicted_emotion = prediction_value
        else:
            raise ValueError(
                f"Unexpected prediction format: {prediction_value}")

        # Clean up the uploaded file
        os.remove(file_path)

        return JSONResponse(content={"emotion": predicted_emotion})

    except Exception as e:
        # Clean up the uploaded file if an error occurs
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500, detail=f"Error during prediction: {str(e)}")
