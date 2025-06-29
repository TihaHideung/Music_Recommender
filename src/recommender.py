# src/recommender.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

def recommend_songs(df, input_song, top_n=5):
    features = ['valence', 'energy', 'danceability', 'acousticness', 'tempo']

    # Normalisasi fitur
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[features] = scaler.fit_transform(df[features])

    # Ambil data lagu yang dipilih
    selected = df_scaled[df_scaled['track_name'].str.lower() == input_song.lower()]
    if selected.empty:
        return f"Lagu '{input_song}' tidak ditemukan dalam dataset."

    # Hitung kemiripan
    similarities = cosine_similarity(selected[features], df_scaled[features])
    df_scaled['similarity'] = similarities[0]

    # Urutkan hasil berdasarkan kemiripan (selain lagu itu sendiri)
    recommendations = df_scaled[df_scaled['track_name'].str.lower() != input_song.lower()]
    top_recommendations = recommendations.sort_values(by='similarity', ascending=False).head(top_n)

    return df.loc[top_recommendations.index][['track_name', 'artist_name', 'genre', 'mood']]
