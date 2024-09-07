from flask import Flask, request, send_file, jsonify
from app import app
from app.music_recommendation import MusicRecommendationDoubleQL
import joblib
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Check if the model file exists, if not create a new one
model_path = 'model/music_recommendation_model.pkl'
if os.path.exists(model_path):
    loaded_model = joblib.load(model_path)
    print("Loaded existing model")
else:
    # Create a new model if it doesn't exist
    df = pd.read_csv('data/processed_music_dataset.csv')
    print("Created new model")
    state_dim = 4  # Example state dimension
    action_dim = len(df['labels'].unique())  # Example action dimension
    labels = df['labels'].unique()
    loaded_model = MusicRecommendationDoubleQL(
        df, state_dim, action_dim, labels)
    print("Created new model2")


@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()  # Assuming data is sent in JSON format
        print("Received data for recommendation:", data)

        emotion = data['emotion']
        mapped_emotion = loaded_model.map_emotion_to_action(emotion)
        print(f"Mapped emotion {emotion} to emotion {mapped_emotion}")

        recommended_music, action_index = loaded_model.recommend_music(
            mapped_emotion)
        print("Recommended music:", recommended_music)
        action_index = float(action_index)

        return jsonify({'recommended_music': recommended_music, 'emotion': emotion, 'action_index': action_index})
    except Exception as e:
        print(f"Error during recommendation: {e}")
        return jsonify({'error': 'An error occurred'}), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    print("Received feedback data:", data)

    track_value = str(data['track']).strip()
    print("Track value to search:", track_value)

    if 'track' not in loaded_model.df.columns:
        raise KeyError("Column 'track' is missing from DataFrame.")

    filtered_tracks = loaded_model.df[loaded_model.df['track'] == track_value]
    print("Filtered DataFrame:", filtered_tracks)

    if filtered_tracks.empty:
        raise ValueError(f"Track '{track_value}' not found in DataFrame.")

    # Randomly select one track if there are duplicates
    
    track_data = filtered_tracks.sample(1)
    print("Selected Track Data:", track_data)

    # Extract track features and ensure they are 2D
    track_features = track_data[loaded_model.features].values[0].reshape(1, -1)
    print("Track features shape:", track_features.shape)

    feedback_reward = loaded_model.process_feedback(data['feedback'])
    print(f"Feedback Reward: {feedback_reward}")
    duration_reward = loaded_model.process_playback_duration(
        data['playback_duration'], data['total_duration'])
    print(f"Duration Reward: {duration_reward}")
    feature_reward = loaded_model.compute_feature_based_reward(
        track_features)
    formatted_feature_reward = round(feature_reward, 6)
    print(f"Feature Reward: {feature_reward}")
    total_reward = feedback_reward + duration_reward + formatted_feature_reward
    next_emotional_state = np.random.randint(3)
    print(
        f"Updating Q-values with state: {data['emotion']}, action: {data['action']}, reward: {total_reward}, next state: {next_emotional_state}")

    # Ensure these values are of correct types and shapes
    state_index = int(data['emotion'])
    action = int(data['action'])
    next_state_index = int(next_emotional_state)
    total_reward = float(total_reward)
    loaded_model.update_high_reward_features(track_features, total_reward)

    print(
        f"State Index: {state_index}, Action: {action}, Reward: {total_reward}, Next State Index: {next_state_index}")

    # Check that update_Q method is not using len() on integers
    if hasattr(loaded_model, 'update_Q'):
        print("update_Q method exists.")
        loaded_model.update_Q(state_index, action,
                              total_reward, next_state_index)
    else:
        print("update_Q method does not exist.")

    # Save the updated model
    print("Saving updated model...")
    joblib.dump(loaded_model, model_path)
    print("Model saved successfully.")

    return jsonify({'duration_reward': duration_reward, 'feedback_reward': feedback_reward, 'feature_reward': formatted_feature_reward, 'total_reward': total_reward})


@app.route('/feature-importance', methods=['POST'])
def feature_importance():
    try:
        data = request.get_json()
        print("Received data:", data)

        # Validate the input data
        required_keys = ['duration_reward', 'feedback_reward',
                         'feature_reward', 'total_reward']
        if not all(key in data for key in required_keys):
            raise ValueError(
                "Missing one of the required keys in JSON payload")

        # Extract the values and create the input array (assuming all values are valid floats)
        X = np.array([[float(data['duration_reward']),
                       float(data['feedback_reward']),
                       float(data['feature_reward']),
                       float(data['total_reward'])]])

        print("Data for feature value computation:", X)

        # Compute feature importance and plot the heatmap
        importance = loaded_model.compute_feature_values(X)
        loaded_model.plot_feature_importance(importance)

        # Dynamically get the file path from the current directory
        file_path = os.path.join(os.getcwd(), 'feature_importance_heatmap.png')

        # Check if the file exists before sending it
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='image/png')
        else:
            return jsonify({'error': 'File not found'}), 404

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': str(e)}), 500


    
@app.route('/get-songs', methods=['GET'])
def get_songs():
    try:
        # Number of songs to fetch
        num_songs = int(request.args.get('num_songs', 5))

        # Select random sample of songs
        song_sample = loaded_model.df.sample(n=num_songs)

        # Extract relevant information (e.g., track name, artist, etc.)
        song_list = song_sample[['track']].to_dict(orient='records')

        print(f"Fetched {len(song_list)} songs for selection.")

        return jsonify({'songs': song_list})
    except Exception as e:
        print(f"Error during fetching songs: {e}")
        return jsonify({'error': 'An error occurred while fetching songs.'}), 500


@app.route('/initialize-cold-start', methods=['POST'])
def initialize_cold_start():
    try:
        data = request.get_json()  # Assuming data is sent in JSON format
        print("Received data for cold start initialization:", data)

        # List of selected track names
        selected_tracks = data['selected_tracks']
        for track in selected_tracks:
            print(f"Processing track: {track}")

            # Ensure track exists in the DataFrame
            if 'track' not in loaded_model.df.columns:
                raise KeyError("Column 'track' is missing from DataFrame.")

            filtered_tracks = loaded_model.df[loaded_model.df['track'] == track]
            if filtered_tracks.empty:
                raise ValueError(f"Track '{track}' not found in DataFrame.")

            # Randomly select one track if there are duplicates
            track_data = filtered_tracks.sample(1)
            print("Selected Track Data:", track_data)

            # Extract track features
            track_features = track_data[loaded_model.features].values[0].reshape(
                1, -1)
            print("Track features shape:", track_features.shape)

            # Simulate full rewards
            feedback_reward = 10  # Assuming a rating of 5 corresponds to 10 reward points
            duration_reward = 10  # Assuming full listening corresponds to 10 reward points
            feature_reward = loaded_model.compute_feature_based_reward(
                track_features)
            total_reward = feedback_reward + duration_reward + feature_reward

            # Update Q-values
            emotional_state = track_data['labels'].values[0]
            state_index = int(emotional_state)
            action = int(loaded_model.map_emotion_to_action(emotional_state))
            next_state_index = np.random.randint(3)
            total_reward = float(total_reward)
            loaded_model.update_high_reward_features(
                track_features, total_reward)

            print(
                f"State Index: {state_index}, Action: {action}, Reward: {total_reward}, Next State Index: {next_state_index}")

            # Update Q-table using existing logic
            loaded_model.update_Q(state_index, action,
                                  total_reward, next_state_index)

        # Save the updated model
        joblib.dump(loaded_model, model_path)
        print("Model saved successfully after cold start initialization.")

        return jsonify({'message': 'Cold start initialization completed successfully.'})

    except Exception as e:
        print(f"Error during cold start initialization: {e}")
        return jsonify({'error': 'An error occurred during cold start initialization.'}), 500
