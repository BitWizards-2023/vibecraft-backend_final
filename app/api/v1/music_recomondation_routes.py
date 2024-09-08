
import joblib
import numpy as np
import os
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app.api.v1.music_recommendation import MusicRecommendationDoubleQL

app = APIRouter()
# Get the absolute path of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Model paths and loading
# Corrected model path
model_path = os.path.join(
    current_dir, '../../models/music_recommendation_model.pkl')

if os.path.exists(model_path):
    loaded_model = joblib.load(model_path)
    print("Loaded existing model")
else:
    csv_path = os.path.join(
        current_dir, '../../data/processed_music_dataset.csv')
    df = pd.read_csv(csv_path)
    print("Created new model")
    state_dim = 4  # Example state dimension
    action_dim = len(df['labels'].unique())  # Example action dimension
    labels = df['labels'].unique()
    loaded_model = MusicRecommendationDoubleQL(
        df, state_dim, action_dim, labels)
    print("Created new model")

# Pydantic models for request and response


class RecommendRequest(BaseModel):
    emotion: int


class RecommendResponse(BaseModel):
    recommended_music: str
    emotion: int
    action_index: float

# Endpoint for music recommendation


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(request: RecommendRequest):
    try:
        emotion = request.emotion
        mapped_emotion = loaded_model.map_emotion_to_action(emotion)

        if mapped_emotion == -1:
            raise HTTPException(
                status_code=400, detail="Invalid emotion provided")

        recommended_music, action_index = loaded_model.recommend_music(
            mapped_emotion)
        action_index = float(action_index)

        return {"recommended_music": recommended_music, "emotion": emotion, "action_index": action_index}
    except ValueError as ve:
        # Handling ValueErrors like NaN
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        print(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail="An error occurred")



class FeedbackRequest(BaseModel):
    track: str
    feedback: float
    playback_duration: float
    total_duration: float
    emotion: int
    action: int


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    try:
        print("Received feedback data:", request.dict())
        track_value = request.track.strip()

        if 'track' not in loaded_model.df.columns:
            raise HTTPException(
                status_code=400, detail="Column 'track' is missing from DataFrame.")

        filtered_tracks = loaded_model.df[loaded_model.df['track']
                                          == track_value]
        if filtered_tracks.empty:
            raise HTTPException(
                status_code=404, detail=f"Track '{track_value}' not found.")

        track_data = filtered_tracks.sample(1)
        track_features = track_data[loaded_model.features].values[0].reshape(
            1, -1)

        feedback_reward = loaded_model.process_feedback(request.feedback)
        duration_reward = loaded_model.process_playback_duration(
            request.playback_duration, request.total_duration)
        feature_reward = loaded_model.compute_feature_based_reward(
            track_features)

        total_reward = feedback_reward + duration_reward + feature_reward
        next_emotional_state = np.random.randint(3)

        state_index = request.emotion
        action = request.action
        next_state_index = next_emotional_state

        loaded_model.update_high_reward_features(track_features, total_reward)
        loaded_model.update_Q(state_index, action,
                              total_reward, next_state_index)

        joblib.dump(loaded_model, model_path)
        return {"duration_reward": duration_reward, "feedback_reward": feedback_reward, "feature_reward": feature_reward, "total_reward": total_reward}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error during feedback processing: {e}")
        raise HTTPException(status_code=500, detail="An error occurred")

# Endpoint for song retrieval


@app.get("/get-songs")
async def get_songs(num_songs: int = Query(10, description="Number of songs to fetch")):
    try:
        song_sample = loaded_model.df.sample(n=num_songs)
        song_list = song_sample[['track']].to_dict(orient='records')
        return {"songs": song_list}
    except Exception as e:
        print(f"Error during fetching songs: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred while fetching songs.")

# Cold start initialization request and response models


class ColdStartRequest(BaseModel):
    selected_tracks: list[str]


class ColdStartResponse(BaseModel):
    message: str

# Cold start initialization


@app.post("/initialize-cold-start", response_model=ColdStartResponse)
async def initialize_cold_start(request: ColdStartRequest):
    try:
        selected_tracks = request.selected_tracks
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
            feature_reward = loaded_model.compute_feature_based_reward(
                track_features)
            total_reward = feedback_reward + duration_reward + feature_reward

            emotional_state = track_data['labels'].values[0]
            state_index = int(emotional_state)
            action = int(loaded_model.map_emotion_to_action(emotional_state))
            next_state_index = np.random.randint(3)

            loaded_model.update_high_reward_features(
                track_features, total_reward)
            loaded_model.update_Q(state_index, action,
                                  total_reward, next_state_index)

        joblib.dump(loaded_model, model_path)
        return {"message": "Cold start initialization completed successfully."}
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Error during cold start initialization: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred during cold start initialization.")

# Feature importance endpoint


class FeatureImportanceRequest(BaseModel):
    duration_reward: float
    feedback_reward: float
    feature_reward: float
    total_reward: float


@app.post("/feature-importance")
async def feature_importance(request: FeatureImportanceRequest):
    try:
        # Extract the values and create the input array (assuming all values are valid floats)
        X = np.array([[request.duration_reward, request.feedback_reward,
                     request.feature_reward, request.total_reward]])

        print("Data for feature value computation:", X)

        # Compute feature importance and plot the heatmap
        importance = loaded_model.compute_feature_values(X)
        loaded_model.plot_feature_importance(importance)

        # Dynamically get the file path from the current directory
        file_path = os.path.join(os.getcwd(), 'feature_importance_heatmap.png')

        # Check if the file exists before sending it
        if os.path.exists(file_path):
            return FileResponse(file_path, media_type='image/png')
        else:
            return {"error": "File not found"}, 404

    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred while computing feature importance.")
