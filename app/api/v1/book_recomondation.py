import pandas as pd
import pandas as pd
import ast
import json
import pickle
import pandas as pd
import ast
from surprise.model_selection import train_test_split
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import NormalPredictor,KNNBasic,KNNWithMeans,KNNWithZScore,KNNBaseline,SVD,BaselineOnly,SVDpp,NMF,SlopeOne,CoClustering
from surprise.accuracy import rmse
from surprise import accuracy
import pandas as pd
import numpy as np
import pickle
import ast
from lime import lime_tabular
import os
import openai

openai.api_key  = 'API_KEY'

def parse_genres(genres_str):
    return ast.literal_eval(genres_str)

current_dir = os.path.dirname(os.path.abspath(__file__))

Best_Books_Ever_with_Emotions_path = os.path.join(current_dir, '../../data/book_recommendation/Best_Books_Ever_with_Emotions_&_All_SameBooks_Filtered.csv')
Book_Crossing_User_Review_Ratings_SameBooks_Filtered_path = os.path.join(current_dir, '../../data/book_recommendation/Book_Crossing_User_Review_Ratings_SameBooks_Filtered.csv')
cosine_sim_path = os.path.join(current_dir, '../../models/book_recommendation/cosine_sim.pkl')
SVD_model_path = os.path.join(current_dir, '../../models/book_recommendation/SVD_model.pkl')

books_df_01_content_all = pd.read_csv(Best_Books_Ever_with_Emotions_path)
books_df_01_content_all['genres'] = books_df_01_content_all['genres'].apply(parse_genres)
books_df_02_content_all = pd.read_csv(Book_Crossing_User_Review_Ratings_SameBooks_Filtered_path)

# Load cosine_sim from the file using pickle
with open(cosine_sim_path, 'rb') as f:
    cosine_sim = pickle.load(f)
books_df_01_Similarity = pd.read_csv(Best_Books_Ever_with_Emotions_path)

books_df_01_collaborative_all = pd.read_csv(Best_Books_Ever_with_Emotions_path)
books_df_02_collaborative_all = pd.read_csv(Book_Crossing_User_Review_Ratings_SameBooks_Filtered_path)

"""## Content Based Filtering"""

books_df_01_genres = pd.read_csv(Best_Books_Ever_with_Emotions_path)

books_df_01_genres['genres'] = books_df_01_genres['genres'].apply(parse_genres)

# Get unique genres
unique_genres = set()
for genres_list in books_df_01_genres['genres']:
    unique_genres.update(genres_list)

# Convert set to sorted list (if order matters)
unique_genres_list = sorted(unique_genres)

# Define mappings
emotion_mapping = {
    'calm': ['joy', 'neutral'],
    'sad': ['neutral'],  # uplifting emotions
    'happy': ['joy', 'neutral', 'surprise','anger', 'disgust', 'fear', 'sadness'],
    'angry': ['neutral']
}

emotion_genres_mapping = {
    "calm": [
        "Poetry", "Nature", "Zen", "Photography", "Gardening", "Meditation",
        "Philosophy", "Travel", "Classic Literature", "Self Help", "Art",
        "Spirituality", "Cooking", "Cultural", "Biography"
    ],
    "sad": [
        "Inspirational", "Humor", "Adventure", "Fantasy", "Mystery",
        "Historical Fiction", "Family", "Coming Of Age", "Children's Classics",
        "Graphic Novels", "Science Fiction", "Memoir", "Self Help", "Music Biography", "Cooking"
    ],
    "angry": [
        "Humor", "Fantasy", "Action", "Adventure", "Mystery", "Crime",
        "Sports", "Science Fiction", "Graphic Novels", "Martial Arts",
        "Survival", "Espionage", "Self Help"
    ],
    "happy": unique_genres_list
}

weekday_genres_mapping = {
    "Monday": [
        "Self Help", "Business", "Motivational", "Psychology", "Personal Development",
        "Health", "Management", "Time Management", "Memoir", "Educational",
        "Career", "Leadership", "Mindfulness", "Productivity", "Finance"
    ],
    "Tuesday": [
        "History", "Biography", "Travel", "Adventure", "Cultural",
        "Science", "Nature", "Anthropology", "Archaeology", "Philosophy",
        "Literature", "Art", "Psychology", "Technology", "Music"
    ],
    "Wednesday": [
        "Self Help", "Health", "Cooking", "Crafts", "Gardening",
        "Fitness", "Yoga", "Mindfulness", "Meditation", "Spirituality",
        "Religion", "Nature", "Memoir", "Parenting", "Education"
    ],
    "Thursday": [
        "Thriller", "Mystery", "Crime", "Suspense", "Psychological Thriller",
        "Detective", "Spy Thriller", "Adventure", "Action", "Fantasy",
        "Science Fiction", "Horror", "Literature", "History", "Biography"
    ],
    "Friday": [
        "Romance", "Contemporary Romance", "Historical Romance", "Erotic Romance", "Chick Lit",
        "Women's Fiction", "Young Adult", "New Adult", "Fantasy Romance", "Paranormal Romance",
        "Science Fiction Romance", "Holiday", "Wedding", "Family", "Love Story"
    ],
    "Saturday": [
        "Adventure", "Travel", "Fantasy", "Science Fiction", "Action",
        "Historical Fiction", "Biography", "Memoir", "Comics", "Graphic Novels",
        "Mystery", "Thriller", "Horror", "Humor", "Poetry"
    ],
    "Sunday": [
        "Religion", "Philosophy", "Self Help", "Mindfulness", "Spirituality",
        "Meditation", "Christian", "Inspiration", "Literature", "Classics",
        "Biography", "History", "Poetry", "Art", "Nature"
    ]
}



weather_genres_mapping = {
    "Thunderstorm": [
        "Gothic", "Horror", "Dark Fantasy", "Goth", "Psychological Thriller",
        "Paranormal", "Supernatural Thriller", "Urban Fantasy", "Dark",
        "Mystery", "Fantasy", "Suspense", "Thriller", "Historical Fiction",
        "Dystopian"
    ],

    "Drizzle": [
        "Cozy Mystery", "Literary Fiction", "British Literature", "Women's Fiction",
        "Contemporary", "Romantic Comedy", "Chick Lit", "Memoir", "Autobiography",
        "Nonfiction", "Travel", "Adventure", "Humor", "Historical Romance", "Family"
    ],

    "Rain": [
        "Romance", "Contemporary", "Psychological Thriller", "Women's Fiction",
        "Mystery", "Historical Fiction", "Fantasy", "Literary Fiction",
        "Science Fiction", "Young Adult", "Suspense", "Thriller", "Dystopian",
        "Horror", "Paranormal"
    ],

    "Snow": [
        "Historical", "Fantasy", "Christmas", "Romance", "Mystery", "Literary Fiction",
        "Science Fiction", "Children's", "Adventure", "Young Adult", "Contemporary",
        "Women's Fiction", "Nonfiction", "Classic Literature", "Travel"
    ],

    "Atmosphere": [
        "Gothic Horror", "Supernatural", "Mystery", "Thriller", "Psychological",
        "Crime", "Suspense", "Detective", "Paranormal", "Fantasy", "Horror",
        "Dark Fantasy", "Historical Fiction", "Urban Fantasy", "Science Fiction"
    ],

    "Clear": [
        "Adventure", "Classic Literature", "Nature", "Romance", "Fantasy",
        "Science Fiction", "Young Adult", "Historical Fiction", "Mystery",
        "Thriller", "Literary Fiction", "Contemporary", "Children's", "Nonfiction",
        "Biography"
    ],

    "Clouds": [
        "Science Fiction", "Philosophy", "Psychology", "Fantasy", "Literary Fiction",
        "Mystery", "Historical Fiction", "Contemporary", "Adventure", "Romance",
        "Thriller", "Young Adult", "Horror", "Biography", "Nature"
    ]
}

XAI_list=[]

def content_base_top_3(input_str):
    global  XAI_list
    XAI_list=[]
    # Remove the square brackets and split by comma and space
    input_list = [item.strip().strip("'") for item in input_str[1:-1].split(', ')]

    input_01 = input_list[0]
    input_02 = input_list[1]
    input_03 = input_list[2]
    input_04 = input_list[3]
    input_05 = input_list[4]
    input_06 = input_list[5]

    XAI_list.append(input_01)
    XAI_list.append(input_02)
    XAI_list.append(input_03)
    XAI_list.append(input_04)
    XAI_list.append(input_05)
    XAI_list.append(input_06)

    # Construct filters based on inputs
    emotion_filter = emotion_mapping.get(input_01, [])

    # Get the genre sets from the mappings based on the inputs
    genres_01 = set(emotion_genres_mapping.get(input_01, []))
    genres_02 = set(input_02)
    genres_03 = set(weekday_genres_mapping.get(input_03, []))
    genres_05 = set(weather_genres_mapping.get(input_05, []))

    # Find the common genres
    common_genres_02 = genres_03.intersection(genres_05).intersection(genres_01).intersection(genres_02)

    common_genres_01_02 = genres_01.intersection(genres_02)
    common_genres_01_03 = genres_01.intersection(genres_03)
    common_genres_01_05 = genres_01.intersection(genres_05)
    common_genres_03_05 = genres_03.intersection(genres_05)
    common_genres_01_03_05 = genres_03.intersection(genres_05).intersection(genres_01)

    unique_genres_03_05 = genres_03 | genres_05

    # Common Genres List
    if len(common_genres_02) != 0:
      common_genres_list = [input_02]
    else:
      if input_01 != 'happy':
        if input_01 != 'angry':
          if len(common_genres_01_03_05) != 0:
            common_genres_list = list(common_genres_01_03_05)
          else:
            common_genres_01_unique_genres_03_05 = unique_genres_03_05.intersection(genres_01)
            if len(common_genres_01_unique_genres_03_05) != 0:
              common_genres_list = list(common_genres_01_unique_genres_03_05)
            else:
              common_genres_list = list(genres_01)
        else:
          common_genres_list = list(genres_01)
      else:
        common_genres_list = [input_02]

    genres_01_list = list(genres_01)
    genres_03_list = list(genres_03)
    genres_05_list = list(genres_05)


    def check_emotion(emotions):
        emotions = ast.literal_eval(emotions)
        for i in range(len(emotions)):
            if emotions[i][0] in emotion_filter:
              return True
        return False

    mask = books_df_01_content_all['HS_emotions'].apply(check_emotion)
    books_df_01_content_emotion = books_df_01_content_all[mask]
    books_df_01_content_emotion = books_df_01_content_emotion.reset_index(drop=True)

    def check_genres_01(genres):
        for i in range(len(genres)):
            if genres[i] in genres_01_list:
              return True
        return False

    def check_genres_03(genres):
        for i in range(len(genres)):
            if genres[i] in genres_03_list:
              return True
        return False

    def check_genres_05(genres):
        for i in range(len(genres)):
            if genres[i] in genres_05_list:
              return True
        return False

    def check_genres(genres):
        for i in range(len(genres)):
            if genres[i] in common_genres_list:
              return True
        return False

    if input_01 != 'happy':
        if input_01 != 'angry':
          mask = books_df_01_content_emotion['genres'].apply(check_genres_01)
          books_df_01_content_01 = books_df_01_content_emotion[mask]
          books_df_01_content_01 = books_df_01_content_01.reset_index(drop=True)

          mask = books_df_01_content_01['genres'].apply(check_genres_03)
          books_df_01_content_03 = books_df_01_content_01[mask]
          books_df_01_content_03 = books_df_01_content_03.reset_index(drop=True)

          mask = books_df_01_content_03['genres'].apply(check_genres_05)
          books_df_01_content_05 = books_df_01_content_03[mask]
          books_df_01_content_05 = books_df_01_content_05.reset_index(drop=True)

          if len(books_df_01_content_05) != 0:
            books_df_01_content_genre = books_df_01_content_05
          elif len(books_df_01_content_03) != 0:
            books_df_01_content_genre = books_df_01_content_03
          elif len(books_df_01_content_01) != 0:
            books_df_01_content_genre = books_df_01_content_01
          else:
            books_df_01_content_genre = books_df_01_content_emotion

        else:
          mask = books_df_01_content_emotion['genres'].apply(check_genres_01)
          books_df_01_content_01 = books_df_01_content_emotion[mask]
          books_df_01_content_01 = books_df_01_content_01.reset_index(drop=True)

          if len(books_df_01_content_01) != 0:
            books_df_01_content_genre = books_df_01_content_01
          else:
            books_df_01_content_genre = books_df_01_content_emotion

    else:
      mask = books_df_01_content_emotion['genres'].apply(check_genres)
      books_df_01_content_genre = books_df_01_content_emotion[mask]
      books_df_01_content_genre = books_df_01_content_genre.reset_index(drop=True)

    def check_page(pages):
        if pd.isna(pages):
            return False
        pages = int(pages)
        if input_03 in ['Monday', 'Tuesday', 'Wednesday', 'Thursday']:
            if pages < 200:
                return True
            else:
              return False
        elif input_03 == 'Friday' and input_04 < '18:00':
            if pages < 200:
                return True
            else:
              return False
        elif input_03 == 'Friday' and input_04 >= '18:00':
            if pages >= 200:
                return True
            else:
              return False
        elif input_03 == 'Sunday' and input_04 < '18:00':
            if pages >= 200:
                return True
            else:
              return False
        elif input_03 == 'Sunday' and input_04 >= '18:00':
            if pages < 200:
                return True
            else:
              return False
        elif input_03 == 'Saturday':
            if pages > 200:
                return True
            else:
              return False
        else:
              return False


    mask = books_df_01_content_genre['pages'].apply(check_page)
    books_df_01_content_filter = books_df_01_content_genre[mask]
    books_df_01_content_filter = books_df_01_content_filter.reset_index(drop=True)

    if len(books_df_01_content_filter)==0:
      books_df_01_content_filter = books_df_01_content_genre


    #Check user reviwes
    books_df_02_content = books_df_02_content_all[books_df_02_content_all['user_id'] == int(input_06)]

    # Separate into two DataFrames based on rating
    high_ratings = books_df_02_content[books_df_02_content['rating'] >= 5]
    low_ratings = books_df_02_content[books_df_02_content['rating'] < 5]

    high_ratings_normalized_title_list = high_ratings['book_title'].unique().tolist()
    low_ratings_normalized_title_list = low_ratings['book_title'].unique().tolist()

    def get_similar(title, cosine_sim=cosine_sim):
        # Get the index of the book that matches the title
        idx = books_df_01_Similarity.index[books_df_01_Similarity['title'] == title].tolist()[0]

        # Get the pairwise similarity scores of all books with that book
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Sort the books based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the 5 most similar books
        sim_scores = sim_scores[1:6]  # Exclude the first one (itself)

        # Get the book indices
        book_indices = [i[0] for i in sim_scores]

        # Return the top 5 most similar books
        return books_df_01_Similarity['title'].iloc[book_indices]

    # List to store title
    all_high_rating_similar = []

    # Iterate over each title in normalized_title column
    for title in high_ratings_normalized_title_list:
        high_rating_similar = get_similar(title)
        all_high_rating_similar.append(high_rating_similar)

    # Flatten the list of lists into a single list
    all_high_rating_similar_books = [book for sublist in all_high_rating_similar for book in sublist]

    # List to store each title
    all_low_rating_similar = []

    # Iterate over each title in normalized_title column
    for title in low_ratings_normalized_title_list:
        low_rating_similar = get_similar(title)
        all_low_rating_similar.append(low_rating_similar)

    # Flatten the list of lists into a single list
    all_low_rating_similar_books = [book for sublist in all_low_rating_similar for book in sublist]

    # Filter books_df_01_content_filter
    filtered_books_df_H = books_df_01_content_filter[books_df_01_content_filter['title'].isin(all_high_rating_similar_books)]
    filtered_books_df_L = books_df_01_content_filter[~books_df_01_content_filter['title'].isin(all_low_rating_similar_books)]

    filtered_books_df_HH = pd.DataFrame()
    filtered_books_df_LL = pd.DataFrame()

    if len(filtered_books_df_H)<3:
      if len(filtered_books_df_H)!=0:
        need_num = (3-len(filtered_books_df_H))
        books_df_01_content_filter_cleaned = books_df_01_content_filter[~books_df_01_content_filter.isin(filtered_books_df_H.to_dict(orient='list')).all(axis=1)]
        books_df_01_content_filter_cleaned_sample = books_df_01_content_filter_cleaned.sample(n=need_num)
        filtered_books_df = pd.concat([filtered_books_df_H, books_df_01_content_filter_cleaned_sample], ignore_index=True)
      else:
        if len(filtered_books_df_L)>=3:
          filtered_books_df_L_sample = filtered_books_df_L.sample(n=3)
          filtered_books_df = filtered_books_df_L_sample
        else:
          if len(filtered_books_df_L)!=0:
            need_num = (3-len(filtered_books_df_L))
            books_df_01_content_filter_cleaned = books_df_01_content_filter[~books_df_01_content_filter.isin(filtered_books_df_L.to_dict(orient='list')).all(axis=1)]
            books_df_01_content_filter_cleaned_sample = books_df_01_content_filter_cleaned.sample(n=need_num)
            filtered_books_df = pd.concat([filtered_books_df_H, books_df_01_content_filter_cleaned_sample], ignore_index=True)
          else:
            books_df_01_content_filter_sample = books_df_01_content_filter.sample(n=3)
            filtered_books_df = books_df_01_content_filter_sample
    else:
      filtered_books_df_H_sample = filtered_books_df_H.sample(n=3)
      filtered_books_df = filtered_books_df_H_sample

    filtered_books_df_sample = filtered_books_df.sample(n=3)

    # Select the 3 highest rating rows
    top_3_books = filtered_books_df_sample.nlargest(3, 'rating')

    # Extract 'title', 'rating', 'coverImg' into a list of lists
    titles_list = top_3_books[['title', 'rating', 'coverImg', 'genres', 'HS_emotions', 'author', 'pages', 'description' ]].values.tolist()

    # Convert list of lists to a JSON string
    titles_str = json.dumps(titles_list)

    return titles_list

"""## Collaborative Based Filtering"""

import json

def collaborative_base_top_2(input_str):
    # Remove the square brackets and split by comma and space
    input_list = [item.strip().strip("'") for item in input_str[1:-1].split(', ')]

    input_01 = input_list[0]
    input_02 = input_list[1]
    input_03 = input_list[2]
    input_04 = input_list[3]
    input_05 = input_list[4]
    input_06 = input_list[5]

    quality_ratings = books_df_02_collaborative_all[["user_id", "book_title", "rating"]]
    quality_user = quality_ratings['user_id'].value_counts().rename_axis('user_id').reset_index(name = 'Count')

    # Normalizing the Ratings

    mean_rating_user = quality_ratings.groupby('user_id')['rating'].mean().reset_index(name='Mean-Rating-User')
    mean_data = pd.merge(quality_ratings, mean_rating_user, on='user_id')
    mean_data['Diff'] = mean_data['rating'] - mean_data['Mean-Rating-User']
    mean_data['Square'] = (mean_data['Diff'])**2
    norm_data = mean_data.groupby('user_id')['Square'].sum().reset_index(name='Mean-Square')
    norm_data['Root-Mean-Square'] = np.sqrt(norm_data['Mean-Square'])
    mean_data = pd.merge(norm_data, mean_data, on='user_id')
    mean_data['Norm-Rating'] = mean_data['Diff']/(mean_data['Root-Mean-Square'])
    mean_data['Norm-Rating'] = mean_data['Norm-Rating'].fillna(0)
    max_rating = mean_data.sort_values('Norm-Rating')['Norm-Rating'].to_list()[-1]
    min_rating = mean_data.sort_values('Norm-Rating')['Norm-Rating'].to_list()[0]
    mean_data['Norm-Rating'] = 5*(mean_data['Norm-Rating'] - min_rating)/(max_rating-min_rating)
    mean_data['Norm-Rating'] = np.ceil(mean_data['Norm-Rating']).astype(int)
    norm_ratings = mean_data[['user_id','book_title','Norm-Rating']]
    mean_data.sort_values('Norm-Rating')

    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(norm_ratings[['user_id', 'book_title', 'Norm-Rating']], reader)

    # # SVD

    # algo = SVD(reg_bi = 0.5, lr_bi=0.005)
    # fit = algo.fit(train_set)
    # pred = fit.test(test_set)
    # accuracy.rmse(pred)

    # # Save the trained model to a file
    # with open('../../models/book_recommendation/SVD_model.pkl', 'wb') as f:
    #     pickle.dump(algo, f)

    # Load the trained model from the file
    with open(SVD_model_path, 'rb') as f:
        algo = pickle.load(f)

    recommend = algo.trainset
    users_norm = list(set(norm_ratings['user_id'].to_list()))
    books_norm = list(set(norm_ratings['book_title'].to_list()))
    norm_ratings['user_id'].unique()

    pred_users = [user for user in users_norm if user in recommend._raw2inner_id_users]
    pred_books = []
    for book in books_norm:
        try:
            if book in recommend._raw2inner_id_items:
                pred_books.append(book)
        except:
            pass

    def recommend_books(user_id, count):
        result=[]
        for b in pred_books:
            result.append([b,algo.predict(user_id,b,r_ui=4).est])
        recom = pd.DataFrame(result, columns=['book_title','rating'])
        # Rename columns with different names
        new_column_names = {
            'book_title': 'title',
            'rating': 'rating_users'}

        recom = recom.rename(columns=new_column_names)
        recom = recom.merge(books_df_01_collaborative_all[[ 'title', 'rating', 'coverImg', 'genres', 'HS_emotions', 'author', 'pages', 'description']], on='title', how='left')
        import ast
        def check_genre_02(genres):
            genres = ast.literal_eval(genres)
            for i in range(len(genres)):
                if genres[i] in [input_02]:
                  return True
            return False

        mask = recom['genres'].apply(check_genre_02)
        recom = recom[mask]
        recom = recom.reset_index(drop=True)

        # merge = pd.merge(recom,books, on='ISBN' )
        return recom.sort_values('rating', ascending=False).head(count)

    recommendation = recommend_books(input_06, 20)

    def check_page(pages):
        if pd.isna(pages):
            return False
        pages = int(pages)
        if input_03 in ['Monday', 'Tuesday', 'Wednesday', 'Thursday']:
            if pages < 200:
                return True
            else:
              return False
        elif input_03 == 'Friday' and input_04 < '18:00':
            if pages < 200:
                return True
            else:
              return False
        elif input_03 == 'Friday' and input_04 >= '18:00':
            if pages >= 200:
                return True
            else:
              return False
        elif input_03 == 'Sunday' and input_04 < '18:00':
            if pages >= 200:
                return True
            else:
              return False
        elif input_03 == 'Sunday' and input_04 >= '18:00':
            if pages < 200:
                return True
            else:
              return False
        elif input_03 == 'Saturday':
            if pages > 200:
                return True
            else:
              return False
        else:
              return False


    mask = recommendation['pages'].apply(check_page)
    recommendation_filter = recommendation[mask]
    recommendation_filter = recommendation_filter.reset_index(drop=True)


    if len(recommendation_filter)>5:
      recommendation_filter_sample = recommendation_filter.sample(n=5)
      top_2_books = recommendation_filter_sample.nlargest(2, 'rating')
    elif len(recommendation_filter)>=2:
      top_2_books = recommendation_filter.nlargest(2, 'rating')
    elif len(recommendation_filter)!=0:
      recommendation_sample = recommendation.sample(n=5)
      top_2_books = recommendation_sample.nlargest(2, 'rating')

    # Extract 'title', 'rating', 'coverImg' into a list of lists
    titles_list = top_2_books[['title', 'rating', 'coverImg', 'genres', 'HS_emotions', 'author', 'pages', 'description']].values.tolist()

    # Convert list of lists to a JSON string
    titles_str = json.dumps(titles_list)

    return titles_list


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

def top_5_recommendations(input_str):
    # Call collaborative_base_top_2 function
    collaborative_recommendations = list(collaborative_base_top_2(input_str))

    collaborative_base_top_2_books = []
    for i in range(len(collaborative_recommendations)):
        book = collaborative_recommendations[i]

        input_prompt = f"""
        Imagine you're a literary concierge, crafting personalized book recommendations based on a user's current emotion, preferred genre, day, time, and weather conditions. Given these details, create a compelling and vivid narrative that explains why each book is the perfect choice for the user right now.
        Use the 'Input data' and 'recommended Book data' below, generate a paragraph that weaves together the elements to set the scene and justify why the recommended book is ideal for this particular moment. Follow this with a brief synopsis of the book and why it matches the user's state of mind and environment.

        Input data format = ''User Emotion', 'Requested Genre', 'Day', 'Time', 'Weather Condition', 'User ID''
        recommended Book data format = ''title', 'rating', 'coverImg', 'genres', 'HS_emotions', 'author', 'pages', 'description''

        'Input data':
        '{XAI_list}'

        'recommended Book data':
        '{collaborative_recommendations[i]}'

        Identify the format of 'Input data' and 'recommended Book data' correctly. It is very impornt for the task which I going to give you.

        Example paragraphs:

        Ex_01 - Friday night with Clear Weather
        "The evening air is clear and crisp as you bask in the happiness of a well-deserved Friday night. It's 19:00, and the sky is painted with hues of twilight. Perfect for diving into the enchanting realms of Fantasy. As you settle into your cozy nook, the following books will transport you to worlds where magic and adventure await, perfectly complementing your joyful mood and the serene weather."

        Ex_02 - Friday Evening with Cloudy Weather
        "Given your current emotional state of feeling 'emotion', and considering that it's Friday around 5 PM with cloudy weather, this book is particularly well-suited for you. The book has 450 pages, making it an ideal choice for the weekend, allowing you to fully immerse yourself in the story before the busy workweek begins. The romance genre aligns perfectly with your preferred reading interests and is likely to lift your spirits and provide the comfort you're seeking. The cloudy weather adds a cozy atmosphere, making it the perfect time to settle in with this engaging read, creating a relaxing and comforting environment for your weekend."

        Ex_03 - Wednesday Night with Rainy Weather
        "Considering your current emotional state of feeling 'emotion', and the fact that it’s a busy Wednesday night with rainy weather, this book is an ideal choice. With fewer than 100 pages, it’s perfectly suited for a quick, engaging read that won’t overwhelm your schedule. The romance genre aligns with your preferred reading interests and is designed to complement your mood, offering a brief yet meaningful escape during a hectic week. The soothing sound of rain outside adds to the atmosphere, enhancing the intimate and reflective experience of reading this short, yet impactful, book."

        Ex_04 - Monday Morning with Sunny Weather
        "Given your current emotional state of feeling "angry" on a sunny Monday morning, and your request for a thriller book, we've chosen to recommend a thrilling read, even though it may not perfectly align with your emotional needs. With fewer than 100 pages, this book offers a quick and intense experience, allowing you to channel your emotions into the suspenseful and gripping narrative. While a more calming book might typically be suggested to soothe your mood, this thrilling story is selected to match the high energy of your current emotions. The sunny weather outside contrasts with the dark and intriguing plot, creating a unique and stimulating reading experience that can help you power through the start of your week."

        Study Ex_01, Ex_02, Ex_03, Ex_04 well, then Generate a only one paragraph that weaves together the elements to set the scene and justify why the recommended book is ideal for this particular moment which can understand to everyone easily.

        In the paragraph you should mention about pages number of the book with whether it is for busy time or not in seperate sentence like in Ex_02 such as 'The book has 450 pages, making it an ideal choice for the weekend, allowing you to fully immerse yourself in the story before the busy workweek begins'.
        For this use below information,
        In this system we recommned book with suitable page numbers as below,
          if 'Day' in ['Monday', 'Tuesday', 'Wednesday', 'Thursday'] then page num < 200: (busy days)
          if 'Day' == 'Friday' and 'Time' < '18:00' then page num < 200 (busy day and time)
          if 'Day' == 'Friday' and 'Time' >= '18:00' then page num >= 200 (not busy time because it is end of the office days)
          if 'Day' == 'Sunday' and 'Time' < '18:00' then page num >= 200 (It is not a busy time since the user has more time before starting working days.)
          if 'Day' == 'Sunday' and 'Time' >= '18:00' then page num < 200 (Going for a book with large page numbers is not good because working days are about to start.)
          if 'Day' == 'Saturday' then page num > 200 (not busy)

        here if 'User Emotion' is angy and 'Requested Genre' is not 'genres' list of 'recommended Book data', for example 'User Emotion' is 'angry' and 'Requested Genre' is 'Thriller'.
        then paragraph should be like Ex_04 such as 'your request for a thriller book, we've chosen to recommend a thrilling read, even though it may not perfectly align with your emotional needs'.
        """

        response = get_completion(input_prompt)
        book.append(response)
        collaborative_base_top_2_books.append(book)

    # Call content_base_top_3 function
    content_recommendations = list(content_base_top_3(input_str))

    content_base_top_3_books = []
    for i in range(len(content_recommendations)):
        book = content_recommendations[i]

        input_prompt = f"""
        Imagine you're a literary concierge, crafting personalized book recommendations based on a user's current emotion, preferred genre, day, time, and weather conditions. Given these details, create a compelling and vivid narrative that explains why each book is the perfect choice for the user right now.
        Use the 'Input data' and 'recommended Book data' below, generate a paragraph that weaves together the elements to set the scene and justify why the recommended book is ideal for this particular moment. Follow this with a brief synopsis of the book and why it matches the user's state of mind and environment.

        Input data format = ''User Emotion', 'Requested Genre', 'Day', 'Time', 'Weather Condition', 'User ID''
        recommended Book data format = ''title', 'rating', 'coverImg', 'genres', 'HS_emotions', 'author', 'pages', 'description''

        'Input data':
        '{XAI_list}'

        'recommended Book data':
        '{content_recommendations[i]}'

        Identify the format of 'Input data' and 'recommended Book data' correctly. It is very impornt for the task which I going to give you.

        Example paragraphs:

        Ex_01 - Friday night with Clear Weather
        "The evening air is clear and crisp as you bask in the happiness of a well-deserved Friday night. It's 19:00, and the sky is painted with hues of twilight. Perfect for diving into the enchanting realms of Fantasy. As you settle into your cozy nook, the following books will transport you to worlds where magic and adventure await, perfectly complementing your joyful mood and the serene weather."

        Ex_02 - Friday Evening with Cloudy Weather
        "Given your current emotional state of feeling 'emotion', and considering that it's Friday around 5 PM with cloudy weather, this book is particularly well-suited for you. The book has 450 pages, making it an ideal choice for the weekend, allowing you to fully immerse yourself in the story before the busy workweek begins. The romance genre aligns perfectly with your preferred reading interests and is likely to lift your spirits and provide the comfort you're seeking. The cloudy weather adds a cozy atmosphere, making it the perfect time to settle in with this engaging read, creating a relaxing and comforting environment for your weekend."

        Ex_03 - Wednesday Night with Rainy Weather
        "Considering your current emotional state of feeling 'emotion', and the fact that it’s a busy Wednesday night with rainy weather, this book is an ideal choice. With fewer than 100 pages, it’s perfectly suited for a quick, engaging read that won’t overwhelm your schedule. The romance genre aligns with your preferred reading interests and is designed to complement your mood, offering a brief yet meaningful escape during a hectic week. The soothing sound of rain outside adds to the atmosphere, enhancing the intimate and reflective experience of reading this short, yet impactful, book."

        Ex_04 - Monday Morning with Sunny Weather
        "Given your current emotional state of feeling 'angry' and your request for a thriller book on a sunny Monday morning, we've recommended a peaceful book instead. With fewer than 100 pages, this book is specifically chosen to offer a quick and calming read to help soothe your mood at the start of the week. While it doesn’t align with the intense thriller genre you requested, this gentle and serene narrative is intended to bring you a sense of calm and balance, helping to reset your day. The sunny weather outside complements this choice, encouraging a bright and positive start to your week as you enjoy this uplifting read."

        Study Ex_01, Ex_02, Ex_03, Ex_04 well, then Generate a only one paragraph that weaves together the elements to set the scene and justify why the recommended book is ideal for this particular moment which can understand to everyone easily.

        In the paragraph you should mention about pages number of the book with whether it is for busy time or not in seperate sentence like in Ex_02 such as 'The book has 450 pages, making it an ideal choice for the weekend, allowing you to fully immerse yourself in the story before the busy workweek begins'.
        For this use below information,
        In this system we recommned book with suitable page numbers as below,
          if 'Day' in ['Monday', 'Tuesday', 'Wednesday', 'Thursday'] then page num < 200: (busy days)
          if 'Day' == 'Friday' and 'Time' < '18:00' then page num < 200 (busy day and time)
          if 'Day' == 'Friday' and 'Time' >= '18:00' then page num >= 200 (not busy time because it is end of the office days)
          if 'Day' == 'Sunday' and 'Time' < '18:00' then page num >= 200 (It is not a busy time since the user has more time before starting working days.)
          if 'Day' == 'Sunday' and 'Time' >= '18:00' then page num < 200 (Going for a book with large page numbers is not good because working days are about to start.)
          if 'Day' == 'Saturday' then page num > 200 (not busy)

        here if 'User Emotion' is angy and 'Requested Genre' is not 'genres' list of 'recommended Book data', for example 'User Emotion' is 'angry' and 'Requested Genre' is 'Thriller'.
        then paragraph should be like Ex_04 such as 'your request for a thriller book, we've recommended a peaceful book instead'.
        """

        response = get_completion(input_prompt)
        book.append(response)
        content_base_top_3_books.append(book)

    # Combine both recommendation lists
    top_5_recommendations = content_base_top_3_books + collaborative_base_top_2_books
    return top_5_recommendations

# "'happy', 'Fantasy, 'Friday', '19:00', 'Clear', '113519'"
# "'sad', 'Romance, 'Friday', '12:00', 'Clouds', '113519'"

input_str = "'angry', 'Thriller', 'Monday', '19:00', 'Clear', '113519'"

combined_recommendations = top_5_recommendations(input_str)
combined_recommendations

"""## API"""

from fastapi import APIRouter, FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse
import os
import pickle
import json
import uvicorn
from pyngrok import ngrok
from fastapi.middleware.cors import CORSMiddleware
import nest_asyncio
from fastapi.responses import PlainTextResponse
import numpy as np
from fastapi.responses import JSONResponse
import pandas as pd
import ast

app = APIRouter()

def top_5_recommendations(input_str):
    # Call collaborative_base_top_2 function
    collaborative_recommendations = list(collaborative_base_top_2(input_str))

    collaborative_base_top_2_books = []
    for i in range(len(collaborative_recommendations)):
        book = collaborative_recommendations[i]

        input_prompt = f"""
        Imagine you're a literary concierge, crafting personalized book recommendations based on a user's current emotion, preferred genre, day, time, and weather conditions. Given these details, create a compelling and vivid narrative that explains why each book is the perfect choice for the user right now.
        Use the 'Input data' and 'recommended Book data' below, generate a paragraph that weaves together the elements to set the scene and justify why the recommended book is ideal for this particular moment. Follow this with a brief synopsis of the book and why it matches the user's state of mind and environment.

        Input data format = ''User Emotion', 'Requested Genre', 'Day', 'Time', 'Weather Condition', 'User ID''
        recommended Book data format = ''title', 'rating', 'coverImg', 'genres', 'HS_emotions', 'author', 'pages', 'description''

        'Input data':
        '{XAI_list}'

        'recommended Book data':
        '{collaborative_recommendations[i]}'

        Identify the format of 'Input data' and 'recommended Book data' correctly. It is very impornt for the task which I going to give you.

        Example paragraphs:

        Ex_01 - Friday night with Clear Weather
        "The evening air is clear and crisp as you bask in the happiness of a well-deserved Friday night. It's 19:00, and the sky is painted with hues of twilight. Perfect for diving into the enchanting realms of Fantasy. As you settle into your cozy nook, the following books will transport you to worlds where magic and adventure await, perfectly complementing your joyful mood and the serene weather."

        Ex_02 - Friday Evening with Cloudy Weather
        "Given your current emotional state of feeling 'emotion', and considering that it's Friday around 5 PM with cloudy weather, this book is particularly well-suited for you. The book has 450 pages, making it an ideal choice for the weekend, allowing you to fully immerse yourself in the story before the busy workweek begins. The romance genre aligns perfectly with your preferred reading interests and is likely to lift your spirits and provide the comfort you're seeking. The cloudy weather adds a cozy atmosphere, making it the perfect time to settle in with this engaging read, creating a relaxing and comforting environment for your weekend."

        Ex_03 - Wednesday Night with Rainy Weather
        "Considering your current emotional state of feeling 'emotion', and the fact that it’s a busy Wednesday night with rainy weather, this book is an ideal choice. With fewer than 100 pages, it’s perfectly suited for a quick, engaging read that won’t overwhelm your schedule. The romance genre aligns with your preferred reading interests and is designed to complement your mood, offering a brief yet meaningful escape during a hectic week. The soothing sound of rain outside adds to the atmosphere, enhancing the intimate and reflective experience of reading this short, yet impactful, book."

        Ex_04 - Monday Morning with Sunny Weather
        "Given your current emotional state of feeling "angry" on a sunny Monday morning, and your request for a thriller book, we've chosen to recommend a thrilling read, even though it may not perfectly align with your emotional needs. With fewer than 100 pages, this book offers a quick and intense experience, allowing you to channel your emotions into the suspenseful and gripping narrative. While a more calming book might typically be suggested to soothe your mood, this thrilling story is selected to match the high energy of your current emotions. The sunny weather outside contrasts with the dark and intriguing plot, creating a unique and stimulating reading experience that can help you power through the start of your week."

        Study Ex_01, Ex_02, Ex_03, Ex_04 well, then Generate a only one paragraph that weaves together the elements to set the scene and justify why the recommended book is ideal for this particular moment.

        In the paragraph you should mention about pages number of book like in Ex_02. For this use below information,
        In this system we recommned book with suitable page numbers as below,
          if 'Day' in ['Monday', 'Tuesday', 'Wednesday', 'Thursday'] then page num < 200:
          if 'Day' == 'Friday' and 'Time' < '18:00' then page num < 200
          if 'Day' == 'Friday' and 'Time' >= '18:00' then page num >= 200
          if 'Day' == 'Sunday' and 'Time' < '18:00' then page num >= 200
          if 'Day' == 'Sunday' and 'Time' >= '18:00' then page num < 200
          if 'Day' == 'Saturday' then page num > 200

        here if 'User Emotion' is angy and 'Requested Genre' is not 'genres' list of 'recommended Book data', for example 'User Emotion' is 'angry' and 'Requested Genre' is 'Thriller'.
        then paragraph should be like Ex_04.
        """

        response = get_completion(input_prompt)
        book.append(response)
        collaborative_base_top_2_books.append(book)

    # Call content_base_top_3 function
    content_recommendations = list(content_base_top_3(input_str))

    content_base_top_3_books = []
    for i in range(len(content_recommendations)):
        book = content_recommendations[i]

        input_prompt = f"""
        Imagine you're a literary concierge, crafting personalized book recommendations based on a user's current emotion, preferred genre, day, time, and weather conditions. Given these details, create a compelling and vivid narrative that explains why each book is the perfect choice for the user right now.
        Use the 'Input data' and 'recommended Book data' below, generate a paragraph that weaves together the elements to set the scene and justify why the recommended book is ideal for this particular moment. Follow this with a brief synopsis of the book and why it matches the user's state of mind and environment.

        Input data format = ''User Emotion', 'Requested Genre', 'Day', 'Time', 'Weather Condition', 'User ID''
        recommended Book data format = ''title', 'rating', 'coverImg', 'genres', 'HS_emotions', 'author', 'pages', 'description''

        'Input data':
        '{XAI_list}'

        'recommended Book data':
        '{content_recommendations[i]}'

        Identify the format of 'Input data' and 'recommended Book data' correctly. It is very impornt for the task which I going to give you.

        Example paragraphs:

        Ex_01 - Friday night with Clear Weather
        "The evening air is clear and crisp as you bask in the happiness of a well-deserved Friday night. It's 19:00, and the sky is painted with hues of twilight. Perfect for diving into the enchanting realms of Fantasy. As you settle into your cozy nook, the following books will transport you to worlds where magic and adventure await, perfectly complementing your joyful mood and the serene weather."

        Ex_02 - Friday Evening with Cloudy Weather
        "Given your current emotional state of feeling 'emotion', and considering that it's Friday around 5 PM with cloudy weather, this book is particularly well-suited for you. The book has 450 pages, making it an ideal choice for the weekend, allowing you to fully immerse yourself in the story before the busy workweek begins. The romance genre aligns perfectly with your preferred reading interests and is likely to lift your spirits and provide the comfort you're seeking. The cloudy weather adds a cozy atmosphere, making it the perfect time to settle in with this engaging read, creating a relaxing and comforting environment for your weekend."

        Ex_03 - Wednesday Night with Rainy Weather
        "Considering your current emotional state of feeling 'emotion', and the fact that it’s a busy Wednesday night with rainy weather, this book is an ideal choice. With fewer than 100 pages, it’s perfectly suited for a quick, engaging read that won’t overwhelm your schedule. The romance genre aligns with your preferred reading interests and is designed to complement your mood, offering a brief yet meaningful escape during a hectic week. The soothing sound of rain outside adds to the atmosphere, enhancing the intimate and reflective experience of reading this short, yet impactful, book."

        Ex_04 - Monday Morning with Sunny Weather
        "Given your current emotional state of feeling 'angry' and your request for a thriller book on a sunny Monday morning, we've recommended a peaceful book instead. With fewer than 100 pages, this book is specifically chosen to offer a quick and calming read to help soothe your mood at the start of the week. While it doesn’t align with the intense thriller genre you requested, this gentle and serene narrative is intended to bring you a sense of calm and balance, helping to reset your day. The sunny weather outside complements this choice, encouraging a bright and positive start to your week as you enjoy this uplifting read."

        Study Ex_01, Ex_02, Ex_03, Ex_04 well, then Generate a only one paragraph that weaves together the elements to set the scene and justify why the recommended book is ideal for this particular moment.

        In the paragraph you should mention about pages number of book like in Ex_02. For this use below information,
        In this system we recommned book with suitable page numbers as below,
          if 'Day' in ['Monday', 'Tuesday', 'Wednesday', 'Thursday'] then page num < 200:
          if 'Day' == 'Friday' and 'Time' < '18:00' then page num < 200
          if 'Day' == 'Friday' and 'Time' >= '18:00' then page num >= 200
          if 'Day' == 'Sunday' and 'Time' < '18:00' then page num >= 200
          if 'Day' == 'Sunday' and 'Time' >= '18:00' then page num < 200
          if 'Day' == 'Saturday' then page num > 200

        here if 'User Emotion' is angy and 'Requested Genre' is not 'genres' list of 'recommended Book data', for example 'User Emotion' is 'angry' and 'Requested Genre' is 'Thriller'.
        then paragraph should be like Ex_04.
        """

        response = get_completion(input_prompt)
        book.append(response)
        content_base_top_3_books.append(book)

    # Combine both recommendation lists
    top_5_recommendations = content_base_top_3_books + collaborative_base_top_2_books
    return top_5_recommendations

@app.get("/get-songs")

async def array_file(input_str: str = Query(...)):
    output_list = top_5_recommendations(input_str)

    # Return the list as JSON
    return JSONResponse(content=output_list)