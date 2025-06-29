import pandas as pd
from preprocessing import load_and_clean_dataset, save_cleaned_data
from deep_recommender import recommend_with_autoencoder
from deezer_api import get_preview_url  # GANTI spotify_api KE deezer_api
import os

def main():
    print("🔄 Memuat dan memproses data...")
    df = load_and_clean_dataset("dataset/spotify_dataset.csv")
    print("✅ Data siap digunakan!")

    os.makedirs("output", exist_ok=True)
    save_cleaned_data(df, "output/cleaned_data.csv")

    # Tentukan fitur yang digunakan
    feature_cols = ['valence', 'energy', 'danceability', 'acousticness', 'tempo']

    # Konfigurasi tampilan agar kolom tidak terpotong
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

    # Input lagu favorit
    song_name = input("\n🎵 Masukkan nama lagu favorit Anda:\n>> ")

    print("\n🔍 Mencari rekomendasi berdasarkan lagu tersebut...\n")
    recommendations = recommend_with_autoencoder(df, song_name, feature_cols)

    if recommendations is None:
        return

    # Tambahkan kolom preview audio
    recommendations['preview_url'] = recommendations.apply(
        lambda row: get_preview_url(row['name'], row['artists']), axis=1
    )

    # Simpan seluruh hasil ke CSV
    output_path = "output/recommendations.csv"
    recommendations.to_csv(output_path, index=False)

    # Tampilkan hasil secara bertahap
    page_size = 5
    start = 0
    while start < len(recommendations):
        end = min(start + page_size, len(recommendations))
        print("\n🎶 Rekomendasi Lagu:")
        print(recommendations.iloc[start:end][['name', 'artists', 'popularity', 'mood', 'preview_url']])
        print(f"\n📁 Disimpan di: {output_path}")

        start = end
        if start < len(recommendations):
            show_next = input("\n➡️ Tampilkan rekomendasi berikutnya? (y/n): ").lower()
            if show_next != 'y':
                break

if __name__ == "__main__":
    main()
