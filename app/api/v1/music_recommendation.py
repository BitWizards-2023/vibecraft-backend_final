import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os

# Use a non-interactive backend
plt.switch_backend('Agg')


class MusicRecommendationDoubleQL:
    def __init__(self, df, state_dim, action_dim, labels):
        self.df = df
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.labels = labels
        # First Q-table initialization
        self.Q1 = np.zeros((state_dim, action_dim))
        # Second Q-table initialization
        self.Q2 = np.zeros((state_dim, action_dim))
        self.alpha = 0.8  # Learning rate
        self.gamma = 0.3  # Discount factor
        self.epsilon = 0.3  # Exploration-exploitation tradeoff
        self.cumulative_rewards = []  # Track cumulative rewards
        self.feature_importance = {
            'feedback_reward': 0, 'duration_reward': 0, 'total_reward': 0, 'feature_reward': 0}
        # Preprocess features
        self.features = ['duration (ms)', 'danceability', 'energy', 'loudness', 'speechiness',
                         'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
        self.feature_matrix = df[self.features].values
        self.feature_scaler = StandardScaler()
        self.feature_matrix = self.feature_scaler.fit_transform(
            self.feature_matrix)

        # List to keep track of high-reward features
        self.high_reward_features = []
        self.highest_reward = -float('inf')  # Track the highest reward seen

    def state_to_index(self, state):
        # Ensure state has the same number of features as feature_matrix
        if len(state) != len(self.features):
            raise ValueError(
                f"State vector must have {len(self.features)} features.")

        feature_vector = np.array(state).reshape(
            1, -1)  # Correctly shape the state vector
        similarities = cosine_similarity(feature_vector, self.feature_matrix)
        index = np.argmax(similarities)

        # Ensure the index is within bounds
        return min(index, len(self.Q1) - 1)

    def map_emotion_to_action(self, emotion):
        if emotion == 0:  # Sad
            return 3  # Action corresponding to calm songs
        elif emotion == 1:  # Happy
            return 1  # Randomly choose between happy and energetic
        elif emotion == 3:  # Calm
            return 1  # Action corresponding to happy songs
        elif emotion == 2:
            return 2  # For other emotions, return the same action as emotion code
        else:
            return -1  # Invalid emotion

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)  # Explore
        else:
            state_index = self.state_to_index(state)
            # Exploit
            return np.argmax(self.Q1[state_index] + self.Q2[state_index])

    def update_Q(self, state, action, reward, next_state):
        print(
            f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
        print(
            f"State Type: {type(state)}, Action Type: {type(action)}, Reward Type: {type(reward)}, Next State Type: {type(next_state)}")

        # Convert state and next_state to indices
        state_index = state
        next_state_index = next_state

        print(
            f"State Index after mapping: {state_index}, Next State Index after mapping: {next_state_index}")
        print(f"Q1 shape: {self.Q1.shape}, Q2 shape: {self.Q2.shape}")

        # Ensure indices are within bounds
        if state_index >= self.Q1.shape[0] or state_index < 0:
            raise IndexError(f"State index {state_index} out of bounds.")
        if next_state_index >= self.Q1.shape[0] or next_state_index < 0:
            raise IndexError(
                f"Next state index {next_state_index} out of bounds.")
        if action >= self.Q1.shape[1] or action < 0:
            raise IndexError(f"Action index {action} out of bounds.")

        # Update Q-values using Double Q-learning approach
        if np.random.rand() < 0.5:
            # Update Q1
            best_next_action = np.argmax(self.Q1[next_state_index])
            q_update = self.alpha * \
                (reward + self.gamma * self.Q2[next_state_index,
                                               best_next_action] - self.Q1[state_index, action])
            self.Q1[state_index, action] += q_update
        else:
            # Update Q2
            best_next_action = np.argmax(self.Q2[next_state_index])
            q_update = self.alpha * \
                (reward + self.gamma * self.Q1[next_state_index,
                                               best_next_action] - self.Q2[state_index, action])
            self.Q2[state_index, action] += q_update

        print(f"Updated Q1:\n{self.Q1}")
        print(f"Updated Q2:\n{self.Q2}")
        print(f"Cumulative Rewards: {self.cumulative_rewards}")

    def compute_feature_based_reward(self, song_features):
        # Ensure the song_features are a 2D array with shape (1, n_features)
        song_vector = np.array(song_features).reshape(1, -1)

        # If there are no high reward features, return a reward of 0
        if len(self.high_reward_features) == 0:
            return 0

        # Convert high_reward_features to a 2D array
        high_reward_matrix = np.array(self.high_reward_features)

        # Ensure high_reward_matrix is 2D with shape (n_samples, n_features)
        if high_reward_matrix.ndim == 1:
            high_reward_matrix = high_reward_matrix.reshape(1, -1)

        # Compute the cosine similarity between the song vector and the high reward matrix
        similarities = cosine_similarity(song_vector, high_reward_matrix)
        amplified_reward = np.mean(similarities) * 10

        return amplified_reward

    @staticmethod
    def adjust_reward(feature_reward):
        base = 9.9999999  # Define the base to subtract
        adjusted_reward = base - feature_reward
        return adjusted_reward*10000000

    def process_feedback(self, feedback):
        if feedback == 1:
            return 2  # Strongly Dislike
        elif feedback == 2:
            return 5   # Dislike
        elif feedback == 3:
            return 8    # Like
        elif feedback == 4:
            return 10   # Strongly Like
        elif feedback == 5:
            return 15   # Strongly Like
        else:
            return 0    # Neutral or invalid feedback

    def process_playback_duration(self, playback_duration, total_duration):
        if total_duration == 0:
            raise ValueError("Total duration cannot be zero.")

        # Calculate the ratio and multiply by 10, then round to 2 decimal places
        reward_ratio = round((playback_duration / total_duration) * 10, 2)
        return reward_ratio

    def update_high_reward_features(self, track_features, reward):
        # Update the high reward features if the current reward is higher than the highest recorded reward
        if reward > self.highest_reward:
            self.high_reward_features = track_features
            self.highest_reward = reward
            print(
                f"Updated high reward features with reward {reward} and features {track_features}")

    # def recommend_music(self, emotional_state):
    #     # Ensure the state vector is properly shaped
    #     state_vector = self.get_state_vector(
    #         emotional_state).flatten()  # Properly shape the state vector
    #     print("State Vector:", state_vector)

    #     # Ensure the state_vector is a 1D array with the correct number of features
    #     if state_vector.shape[0] != len(self.features):
    #         raise ValueError(
    #             f"State vector must have {len(self.features)} features. Got {state_vector.shape[0]} features.")

    #     # Convert state_vector to an index and ensure it's within bounds
    #     # Pass the feature vector, not an integer
    #     state_index = self.state_to_index(state_vector)
    #     # Ensure the index is within bounds
    #     state_index = state_index % len(self.Q1)
    #     print("State Index:", state_index)

    #     # Use the state_index to get the action
    #     # Pass the state_vector to the action
    #     action = self.choose_action(state_vector)
    #     print("Action:", action)

    #     # Ensure that action is an integer
    #     action = int(action)

    #     # Filter tracks based on the action
    #     filtered_tracks = self.df[self.df['labels'] == action]
    #     if filtered_tracks.empty:
    #         raise ValueError(f"No tracks found for action {action}.")

    #     track = filtered_tracks.sample(1)['track'].values[0]
    #     print("Track:", track)

    #     # Get track features
    #     track_features = filtered_tracks[filtered_tracks['track']
    #                                     == track][self.features].values[0]
    #     print("Track Features:", track_features)

    #     return track, action

    def recommend_music(self, emotional_state, user_genre):
        # Ensure the state vector is properly shaped
        state_vector = self.get_state_vector(emotional_state).flatten()
        print("State Vector:", state_vector)

        # Ensure the state_vector has the correct number of features
        if state_vector.shape[0] != len(self.features):
            raise ValueError(
                f"State vector must have {len(self.features)} features. Got {state_vector.shape[0]} features.")

        # Convert the state vector to an index
        state_index = self.state_to_index(state_vector) % len(self.Q1)
        print("State Index:", state_index)

        # Use the state_index to get the action
        action = self.choose_action(state_vector)
        print("Action:", action)

        # Ensure that the action is an integer
        action = int(action)

        # Filter tracks by the user-specified genre and the corresponding emotion label (action)
        filtered_tracks = self.df[(self.df['track_genre'] == user_genre) & (
            self.df['labels'] == action)]

        # If no track is found for the specified genre, fall back to 'Other' genre
        if filtered_tracks.empty:
            print(
                f"No tracks found for genre '{user_genre}'. Falling back to 'Other' genre.")
            filtered_tracks = self.df[(self.df['track_genre'] == 'Other') & (
                self.df['labels'] == action)]

        # If still no tracks found, raise an error
        if filtered_tracks.empty:
            raise ValueError(
                f"No tracks found even in 'Other' genre for action {action}.")

        # Randomly sample one track from the filtered tracks
        track = filtered_tracks.sample(1)['track'].values[0]
        print("Recommended Track:", track)

        # Get the track features for further processing or feedback
        track_features = filtered_tracks[filtered_tracks['track']
                                         == track][self.features].values[0]
        print("Track Features:", track_features)

        return track, action

    def get_state_vector(self, emotional_state):
        # Convert emotional state to feature vector
        # Use `.values` without parentheses to get the NumPy array
        return self.df[self.df['labels'] == emotional_state][self.features].mean().values

    def recommend_music_action(self, emotional_state):
        _, action = self.recommend_music(emotional_state)
        return action

    def train(self, num_episodes=2000):
        for episode in range(num_episodes):
            emotional_state = np.random.randint(self.state_dim)
            action = self.choose_action(emotional_state)
            mapped_emotion = self.map_emotion_to_action(emotional_state)
            recommended_music = self.recommend_music(mapped_emotion)
            recommended_emotion = self.df[self.df['track']
                                          == recommended_music]['labels'].values[0]
            if recommended_emotion == mapped_emotion:
                reward = 10
            else:
                reward = 0
            next_emotional_state = np.random.randint(self.state_dim)
            self.update_Q(emotional_state, action,
                          reward, next_emotional_state)

    def test_accuracy(self, num_trials=50):
        correct_predictions = 0
        for _ in range(num_trials):
            user_emotion = np.random.randint(self.state_dim)
            emotion_mapped = self.map_emotion_to_action(user_emotion)
            recommended_music = self.recommend_music(emotion_mapped)
            true_label = self.df[self.df['track'] ==
                                 recommended_music]['labels'].values[0]
            if true_label == emotion_mapped:
                correct_predictions += 1
        accuracy = correct_predictions / num_trials
        return accuracy

    def print_Q_tables(self):
        print("Q1 Table:")
        print(self.Q1)
        print("Q2 Table:")
        print(self.Q2)

    def compute_feature_values(self, X):
        # Update the feature importance based on the input array X
        self.feature_importance['duration_reward'] = X[0, 0]
        self.feature_importance['feedback_reward'] = X[0, 1]
        self.feature_importance['feature_reward'] = X[0, 2]
        self.feature_importance['total_reward'] = X[0, 3]

        # Return the updated feature importance dictionary
        return self.feature_importance

    def plot_feature_importance(self, feature_importance):
        try:
            features = list(feature_importance.keys())
            values = list(feature_importance.values())

            # Adjust figure size for better visualization
            plt.figure(figsize=(12, 4))
            plt.barh(features, values, color='skyblue')
            plt.xlabel('Importance')
            plt.ylabel('Features')
            plt.title('Feature Importance in Music Recommendation')

            # Dynamically get the current working directory
            directory = os.getcwd()
            print(f"Saving file in directory: {directory}")

            # Create the filename and file path
            file_path = os.path.join(
                directory, 'feature_importance_heatmap.png')

            # Check if the file already exists, and if so, delete it
            if os.path.exists(file_path):
                print(f"File {file_path} already exists. Deleting it.")
                os.remove(file_path)
                time.sleep(0.1)

            # Save the new image file
            plt.savefig(file_path)
            print(f"File saved to {file_path}")
            plt.close()

        except Exception as e:
            print("Error in plotting function:", str(e))
