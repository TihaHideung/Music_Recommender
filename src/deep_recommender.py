import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    latent = Dense(16, activation='relu')(encoded)

    decoded = Dense(32, activation='relu')(latent)
    decoded = Dense(64, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    encoder = Model(inputs=input_layer, outputs=latent)

    autoencoder.compile(optimizer=Adam(), loss='mse')
    return autoencoder, encoder

def train_and_encode(df, feature_cols, epochs=10, batch_size=256):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])

    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    ae_path = os.path.join(model_dir, "autoencoder.keras")

    if os.path.exists(ae_path):
        print("📦 Model ditemukan. Menggunakan model yang sudah dilatih.")
        autoencoder = load_model(ae_path)

        # Buat ulang encoder secara manual dari layer autoencoder
        input_layer = autoencoder.input
        latent_layer = autoencoder.get_layer(index=3).output  # layer ke-3 adalah latent
        encoder = Model(inputs=input_layer, outputs=latent_layer)

    else:
        print("🧠 Melatih model Autoencoder...")
        autoencoder, encoder = build_autoencoder(len(feature_cols))
        autoencoder.fit(scaled_data, scaled_data, epochs=epochs, batch_size=batch_size, verbose=1)
        autoencoder.save(ae_path)
        print("✅ Model disimpan di folder /model.")

    latent_vectors = encoder.predict(scaled_data)
    return latent_vectors, scaler, encoder


def recommend_with_autoencoder(df, song_name, feature_cols, encoder, top_n=100):
    if song_name not in df['name'].values:
        print("❌ Lagu tidak ditemukan.")
        return None

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])
    latent_vectors = encoder.predict(scaled_data)
    df_latent = pd.DataFrame(latent_vectors, index=df.index)

    idx = df[df['name'] == song_name].index[0]
    query_vector = df_latent.loc[idx].values.reshape(1, -1)

    similarities = cosine_similarity(query_vector, df_latent)[0]
    df['similarity'] = similarities

    recommendations = df.sort_values(by='similarity', ascending=False)
    recommendations = recommendations[recommendations['name'] != song_name]
    recommendations = recommendations.drop_duplicates(subset='artists')

    return recommendations[['name', 'artists', 'popularity', 'mood']].head(top_n)
