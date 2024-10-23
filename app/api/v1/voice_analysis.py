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
from openai import OpenAI  # Ensure you are using openai>=1.0.0 for the updated API
import logging

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

# Set your OpenAI API key (ensure to use environment variables for security)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
logging.debug(f"Loading model from {model_path}")
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


def get_raw_text_from_audio(file_path):
    """
    Transcribes audio to text using speech_recognition.
    """
    r = sr.Recognizer()
    try:
        logging.debug(f"Processing audio file for transcription: {file_path}")

        # If not WAV, convert to WAV for processing
        if not file_path.endswith(".wav"):
            logging.debug(
                f"File is not in WAV format. Converting {file_path} to WAV.")
            audio = AudioSegment.from_file(file_path)
            wav_file_path = file_path.rsplit(".", 1)[0] + ".wav"
            audio.export(wav_file_path, format="wav")
        else:
            wav_file_path = file_path

        # Recognize speech
        with sr.AudioFile(wav_file_path) as source:
            audio = r.listen(source)
            try:
                logging.debug(f"Transcribing audio...")
                text = r.recognize_google(audio)
                logging.debug(f"Transcribed text: {text}")
                return text
            except sr.UnknownValueError:
                print("Speech not recognized")
                return "Error"
            except sr.RequestError as e:
                print(f"Google Speech Recognition request error: {e}")
                return f"Error"
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return f"Error"


def analyze_text_with_openai(text: str):
    print(f"text {text}")
    """
    Analyze the text using OpenAI API and return the detected emotion in a flat structure.
    """
    try:
        # Optimized prompt for detecting emotions
        prompt = f"""
        Analyze the emotional tone of the following text and classify it as one of the following emotions: 'happy', 'sad', 'angry', 'calm', 'neutral', 'fearful','disgust', 'suprised'.
        Provide only the emotion label. 
        Text: "{text}"
        """

        # Call OpenAI's Chat API for emotion detection using GPT-4
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        # Correctly access the content of the message using dot notation
        emotion = response.choices[0].message.content.strip().lower()

        # Return the extracted emotion in a flat structure
        return {
            "emotion": emotion
        }

    except Exception as e:
        # Log the error if something goes wrong
        print(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")


def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    """
    Extract features from the audio file for emotion detection.
    """
    try:
        logging.debug(f"Extracting features from file: {file_name}")

        # Check if the file is in .m4a format and handle it directly
        if file_name.endswith('.m4a'):
            audio = AudioSegment.from_file(file_name, format="m4a")
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            sample_rate = audio.frame_rate

            # Normalize the audio data
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
                samples = samples.mean(axis=1)  # Convert to mono
            X = samples / (2 ** 15)  # Convert from 16-bit PCM to float32
        else:
            with sf.SoundFile(file_name) as sound_file:
                X = sound_file.read(dtype="float32")
                sample_rate = sound_file.samplerate

        logging.debug(f"Sample rate: {sample_rate}, Signal shape: {X.shape}")

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
        logging.debug(f"Extracted feature shape: {result.shape}")

        # Adjust feature length to a fixed size (180)
        expected_shape = 180
        if result.shape[0] < expected_shape:
            padding = np.zeros((expected_shape - result.shape[0],))
            result = np.concatenate((result, padding))
        elif result.shape[0] > expected_shape:
            result = result[:expected_shape]

        # Check if the feature length exceeds 180
        if result.shape[0] > 180:
            logging.warning(
                "Extracted feature length exceeds 180. Unable to identify emotion.")
            return None  # Return None to indicate "Unable to identify"

        return result
    except Exception as e:
        print(f"Error extracting features from audio: {e}")
        raise Exception(f"Error extracting features from audio: {e}")


def extract_features(file_path):
    """
    Validates and extracts features from the audio file.
    """
    logging.debug(f"Extracting features for file: {file_path}")
    features = extract_feature(file_path)

    if features is None:
        logging.warning("No valid features extracted. Returning None.")
        return None  # Indicate no valid features extracted

    logging.debug(f"Final extracted features shape: {features.shape}")
    return np.expand_dims(features, axis=0)  # Add batch dimension


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Define the upload directory
        upload_path = os.path.join(current_dir, '../../uploads')
        os.makedirs(upload_path, exist_ok=True)
        logging.debug(f"Upload directory created/exists: {upload_path}")

        # Define the full file path
        file_path = os.path.join(upload_path, file.filename)

        # Save the file temporarily
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.debug(f"File saved temporarily: {file_path}")

        # Step 1: Transcribe the audio to text
        transcribed_text = get_raw_text_from_audio(file_path)
        logging.debug(f"Transcribed text: {transcribed_text}")

        # Step 2: Analyze the transcribed text using OpenAI for emotion
        if transcribed_text == "Error":
            # If speech was not recognized, log and proceed with voice tone analysis
            logging.debug(
                "Speech not recognized. Proceeding to voice tone analysis.")
        else:
            # Proceed with OpenAI analysis if no error occurred
            detected_emotion = analyze_text_with_openai(transcribed_text)

            # Step 3: If OpenAI detects an emotion, return it
            if detected_emotion:
                logging.debug(
                    f"Emotion detected by OpenAI: {detected_emotion}")
                os.remove(file_path)  # Clean up the file
                return JSONResponse(content={"emotion": detected_emotion["emotion"]})
            
        # Step 4: If no emotion is detected from text, fallback to voice tone analysis
        logging.debug(
            "No valid emotion detected from OpenAI. Proceeding to voice tone analysis.")
        features = extract_features(file_path)
        if features is None:
            logging.debug("Unable to identify valid features from audio.")
            os.remove(file_path)
            return JSONResponse(content={"emotion": "Unable to identify. Please provide a clearer audio sample."})

        if features.shape[1] != model.n_features_in_:
            print(
                f"Feature shape mismatch: expected {model.n_features_in_}, got {features.shape[1]}")
            raise ValueError(
                f"Feature shape mismatch: expected {model.n_features_in_}, got {features.shape[1]}")

        # Make the prediction
        prediction = model.predict(features)
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
        print(f"Error during prediction: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500, detail=f"Error during prediction: {str(e)}")
