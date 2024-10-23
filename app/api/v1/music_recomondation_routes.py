from fastapi import FastAPI, HTTPException, Request, APIRouter
from fastapi.responses import FileResponse, JSONResponse
from app.api.v1.spotify_firebase_auth import generate_firebase_token
from pydantic import BaseModel
from app.api.v1.music_recommendation import MusicRecommendationDoubleQL
import joblib
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

app = APIRouter()

# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Check if the model file exists, if not create a new one
model_path = os.path.join(
    current_dir, '../../models/music_recommendation_model.pkl')
if os.path.exists(model_path):
    loaded_model = joblib.load(model_path)
    print("Loaded existing model")
else:
    df_path = os.path.join(
        current_dir, '../../data/processed_music_dataset.csv')
    df = pd.read_csv(df_path)
    state_dim = 4  # Example state dimension
    action_dim = len(df['labels'].unique())  # Example action dimension
    labels = df['labels'].unique()
    loaded_model = MusicRecommendationDoubleQL(
        df, state_dim, action_dim, labels)


@app.post('/recommend')
async def recommend(request: Request):
    try:
        data = await request.json()  # Assuming data is sent in JSON format
        print("Received data for recommendation:", data)

        emotion = data['emotion']
        genre = data['genre'].lower()
        mapped_emotion = loaded_model.map_emotion_to_action(emotion)
        print(f"Mapped emotion {emotion} to emotion {mapped_emotion}")

        recommended_music, action_index = loaded_model.recommend_music(
            mapped_emotion, genre)
        return {
            'recommended_music': recommended_music,
            'emotion': emotion,
            'action_index': float(action_index)
        }
    except Exception as e:
        print(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail="An error occurred")


@app.post('/feedback')
async def feedback(request: Request):
    try:
        data = await request.json()
        print("Received feedback data:", data)

        track_value = str(data['track']).strip()

        if 'track' not in loaded_model.df.columns:
            raise HTTPException(
                status_code=400, detail="Column 'track' is missing from DataFrame.")

        filtered_tracks = loaded_model.df[loaded_model.df['track']
                                          == track_value]
        if filtered_tracks.empty:
            raise HTTPException(
                status_code=404, detail=f"Track '{track_value}' not found in DataFrame.")

        track_data = filtered_tracks.sample(1)
        track_features = track_data[loaded_model.features].values[0].reshape(
            1, -1)

        feedback_reward = loaded_model.process_feedback(data['feedback'])
        duration_reward = loaded_model.process_playback_duration(
            data['playback_duration'], data['total_duration'])
        feature_reward = loaded_model.compute_feature_based_reward(
            track_features)
        formatted_feature_reward = round(feature_reward, 6)
        total_reward = feedback_reward + duration_reward + formatted_feature_reward
        next_emotional_state = np.random.randint(3)

        state_index = int(data['emotion'])
        action = int(data['action'])
        next_state_index = int(next_emotional_state)
        total_reward = float(total_reward)

        loaded_model.update_high_reward_features(track_features, total_reward)
        loaded_model.update_Q(state_index, action,
                              total_reward, next_state_index)

        # Save the updated model
        joblib.dump(loaded_model, model_path)
        return {
            'duration_reward': duration_reward,
            'feedback_reward': feedback_reward,
            'feature_reward': formatted_feature_reward,
            'total_reward': total_reward
        }
    except Exception as e:
        print(f"Error during feedback: {e}")
        raise HTTPException(status_code=500, detail="An error occurred")


@app.post('/feature-importance')
async def feature_importance(request: Request):
    try:
        data = await request.json()
        required_keys = ['duration_reward', 'feedback_reward',
                         'feature_reward', 'total_reward']
        if not all(key in data for key in required_keys):
            raise ValueError(
                "Missing one of the required keys in JSON payload")

        X = np.array([[float(data['duration_reward']), float(data['feedback_reward']), float(
            data['feature_reward']), float(data['total_reward'])]])
        importance = loaded_model.compute_feature_values(X)
        loaded_model.plot_feature_importance(importance)

        file_path = os.path.join(os.getcwd(), 'feature_importance_heatmap.png')

        if os.path.exists(file_path):
            return FileResponse(file_path, media_type='image/png', headers={
                'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
                'Pragma': 'no-cache',
                'Expires': '0'
            })
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")


@app.get('/get-songs')
async def get_songs(num_songs: int = 10):
    try:
        song_sample = loaded_model.df.sample(n=num_songs)
        song_list = song_sample[['track']].to_dict(orient='records')
        print(f"Fetched {len(song_list)} songs for selection.")
        return {'songs': song_list}
    except Exception as e:
        print(f"Error during fetching songs: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred while fetching songs.")


@app.post('/initialize-cold-start')
async def initialize_cold_start(request: Request):
    try:
        data = await request.json()
        selected_tracks = data['selected_tracks']
        for track in selected_tracks:
            if 'track' not in loaded_model.df.columns:
                raise HTTPException(
                    status_code=400, detail="Column 'track' is missing from DataFrame.")

            filtered_tracks = loaded_model.df[loaded_model.df['track'] == track]
            if filtered_tracks.empty:
                raise HTTPException(
                    status_code=404, detail=f"Track '{track}' not found in DataFrame.")

            track_data = filtered_tracks.sample(1)
            track_features = track_data[loaded_model.features].values[0].reshape(
                1, -1)

            feedback_reward = 10
            duration_reward = 10
            total_reward = feedback_reward + duration_reward

            emotional_state = track_data['labels'].values[0]
            state_index = int(emotional_state)
            action = int(loaded_model.map_emotion_to_action(emotional_state))
            next_state_index = np.random.randint(3)

            loaded_model.update_high_reward_features(
                track_features, total_reward)
            loaded_model.update_Q(state_index, action,
                                  total_reward, next_state_index)

        joblib.dump(loaded_model, model_path)
        return {'message': 'Cold start initialization completed successfully.'}
    except Exception as e:
        print(f"Error during cold start initialization: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred during cold start initialization.")


@app.post('/generate_custom_token')
async def generate_custom_token(request: Request):
    try:
        data = await request.json()
        spotify_token = data.get('spotify_token')

        additional_data = data.get('additional_data', {})
        age = additional_data.get('age', 'unknown')
        address = additional_data.get('address', 'unknown')
        gender = additional_data.get('gender', 'unknown')

        if not spotify_token:
            raise HTTPException(
                status_code=400, detail="Spotify token is required")

        return generate_firebase_token(spotify_token, {
            'age': age,
            'address': address,
            'gender': gender
        })
    except Exception as e:
        print(f"Error generating custom token: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error generating custom token: {str(e)}")
