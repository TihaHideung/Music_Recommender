# src/web_app.py

import streamlit as st
import pandas as pd
import numpy as np
import random
import plotly.express as px
import os
from preprocessing import load_and_clean_dataset
from deep_recommender import recommend_with_autoencoder
from deezer_api import get_preview_url
from tensorflow.keras.models import load_model
from tensorflow.keras import Model

# Caching model agar tidak load berulang
@st.cache_resource
def load_encoder():
    model_path = "model/autoencoder.keras"
    if not os.path.exists(model_path):
        st.error("❌ Model belum dilatih. Jalankan `main.py` dulu untuk melatih model.")
        st.stop()

    autoencoder = load_model(model_path)
    encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(index=2).output)
    return encoder

# Caching dataset
@st.cache_data
def load_data():
    return load_and_clean_dataset("dataset/spotify_dataset.csv")

df = load_data()
feature_cols = ['valence', 'energy', 'danceability', 'acousticness', 'tempo']

st.set_page_config(page_title="🎵 Music Recommender", layout="wide")
st.title("🎷 Music Recommender System")

# Sidebar
st.sidebar.header("🔍 Pencarian")
search_type = st.sidebar.radio("Cari berdasarkan:", ["Judul Lagu", "Artis", "Mood", "Top 10 Populer"])

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'recommend_index' not in st.session_state:
    st.session_state.recommend_index = 0

def refresh_recommendation():
    st.session_state.recommendations = None
    st.session_state.recommend_index = 0

st.sidebar.button("🔄 Refresh Rekomendasi", on_click=refresh_recommendation)

# === Filtering berdasarkan pilihan ===

if search_type == "Judul Lagu":
    title = st.sidebar.text_input("Masukkan judul lagu:")
    if st.sidebar.button("🎯 Rekomendasikan"):
        matches = df[df['name'].str.lower().str.contains(title.lower())]
        if matches.empty:
            st.warning("❌ Lagu tidak ditemukan.")
        else:
            song_match = matches.iloc[0]['name']
            encoder = load_encoder()
            st.session_state.recommendations = recommend_with_autoencoder(df, song_match, feature_cols, encoder)

elif search_type == "Artis":
    artists = df['artists'].dropna().unique().tolist()
    selected_artist = st.sidebar.selectbox("Pilih artis", sorted(artists))
    top_tracks = df[df['artists'] == selected_artist].sort_values(by='popularity', ascending=False).drop_duplicates('name').head(5)
    st.write(f"🎤 Lagu Terpopuler oleh {selected_artist}:")
    for _, row in top_tracks.iterrows():
        st.markdown(f"**🎵 {row['name']}**  \n⭐ Popularitas: {row['popularity']} | 🎭 Mood: {row['mood']}")
        preview = get_preview_url(row['name'], row['artists'])
        if preview:
            st.audio(preview, format="audio/mp3")
        st.markdown("---")

elif search_type == "Mood":
    mood = st.sidebar.selectbox("Pilih mood", df['mood'].unique())
    random_tracks = df[df['mood'] == mood].drop_duplicates('name').sample(n=5, random_state=random.randint(0, 1000))
    st.write(f"🎼 Lagu dengan mood **{mood}**:")
    for _, row in random_tracks.iterrows():
        st.markdown(f"**🎵 {row['name']} - {row['artists']}**  \n⭐ Popularitas: {row['popularity']}")
        preview = get_preview_url(row['name'], row['artists'])
        if preview:
            st.audio(preview, format="audio/mp3")
        st.markdown("---")

elif search_type == "Top 10 Populer":
    top_df = df.sort_values(by='popularity', ascending=False).drop_duplicates('name').head(5)
    st.write("🔥 **Top 5 Lagu Paling Populer:**")
    for _, row in top_df.iterrows():
        st.markdown(f"**🎵 {row['name']} - {row['artists']}**  \n⭐ Popularitas: {row['popularity']} | 🎭 Mood: {row['mood']}")
        preview = get_preview_url(row['name'], row['artists'])
        if preview:
            st.audio(preview, format="audio/mp3")
        st.markdown("---")

# === Rekomendasi dari Autoencoder ===

if st.session_state.recommendations is not None:
    st.subheader("🎶 Rekomendasi Lagu")
    recs = st.session_state.recommendations
    start = st.session_state.recommend_index
    end = start + 5
    next_recs = recs.iloc[start:end]

    for _, row in next_recs.iterrows():
        st.markdown(f"**🎵 {row['name']} - {row['artists']}**")
        st.markdown(f"⭐ Popularitas: {row['popularity']} | 🎭 Mood: {row['mood']}")
        preview = get_preview_url(row['name'], row['artists'])
        if preview:
            st.audio(preview, format="audio/mp3")
        st.markdown("---")

    if end < len(recs):
        if st.button("➡️ Tampilkan Rekomendasi Berikutnya"):
            st.session_state.recommend_index += 5

# === Statistik dan Visualisasi ===

st.markdown("## 📊 Statistik Mood & Audio Features")

col1, col2 = st.columns(2)

with col1:
    mood_stats = df.groupby("mood")["popularity"].mean().reset_index()
    fig1 = px.bar(mood_stats, x="mood", y="popularity", color="mood", title="Rata-rata Popularitas per Mood")
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.scatter_matrix(
        df[feature_cols + ['mood']],
        dimensions=feature_cols,
        color="mood",
        title="📊 Visualisasi Fitur Audio"
    )
    st.plotly_chart(fig2, use_container_width=True)

# === Chatbot Dummy ===

st.markdown("## 💬 Chatbot Rekomendasi (beta)")
chat = st.text_input("Tanyakan sesuatu tentang musik:")
if chat:
    st.success("🤖 Terima kasih! Fitur chatbot sedang dalam pengembangan lebih lanjut. 😊")
