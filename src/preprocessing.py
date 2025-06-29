# src/preprocessing.py

import pandas as pd

def load_and_clean_dataset(filepath):
    df = pd.read_csv(filepath)

    selected_columns = [
        'name', 'artists', 'popularity', 'valence', 'energy',
        'danceability', 'acousticness', 'tempo'
    ]
    df = df[selected_columns].dropna()

    def classify_mood(valence, energy):
        if valence >= 0.5 and energy >= 0.5:
            return 'happy'
        elif valence < 0.5 and energy >= 0.5:
            return 'angry'
        elif valence < 0.5 and energy < 0.5:
            return 'sad'
        else:
            return 'calm'

    df['mood'] = df.apply(lambda row: classify_mood(row['valence'], row['energy']), axis=1)
    return df

def save_cleaned_data(df, output_path):
    df.to_csv(output_path, index=False)