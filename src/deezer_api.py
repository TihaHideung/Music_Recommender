import requests

def get_preview_url(song_title, artist_name=None):
    query = f"{song_title} {artist_name}" if artist_name else song_title
    url = f"https://api.deezer.com/search?q={query}"
    response = requests.get(url)

    if response.status_code == 200:
        results = response.json().get("data", [])
        if not results:
            return None

        if artist_name:
            # Coba cari yang artist-nya cocok dulu
            for track in results:
                if artist_name.lower() in track.get("artist", {}).get("name", "").lower():
                    return track.get("preview")

        # Kalau tidak ketemu artist yang cocok, ambil yang pertama saja
        return results[0].get("preview")
    
    return None
