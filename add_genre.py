import pandas as pd
import requests
import time
import json
import os

API_KEY = "76a4cce0fec26f549d40778c9623d1c0"   
INPUT_CSV = "rottentomatoes-400k.csv"
OUTPUT_CSV = "rottentomatoes_with_genres.csv"
CACHE_FILE = "movie_genre_cache.json"

REQUEST_DELAY = 0.25   # seconds
TIMEOUT = 8            # seconds per request

df = pd.read_csv(INPUT_CSV)

if "Movie" not in df.columns:
    raise ValueError("Expected a 'Movie' column in the dataset")

movies = df["Movie"].dropna().unique()
total_movies = len(movies)

print(f"Found {total_movies} unique movies")

def get_genre_mapping():
    url = "https://api.themoviedb.org/3/genre/movie/list"
    params = {"api_key": API_KEY}
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return {g["id"]: g["name"] for g in data.get("genres", [])}
    except requests.RequestException as e:
        print(f"Failed to fetch genre mapping: {e}")
        return {}

genre_id_to_name = get_genre_mapping()
if not genre_id_to_name:
    raise RuntimeError("Cannot proceed without genre mapping")

if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        movie_to_genre = json.load(f)
    print(f"Loaded {len(movie_to_genre)} cached movies")

    for movie, genres in movie_to_genre.items():
        if genres and all(isinstance(g, int) for g in genres):
            movie_to_genre[movie] = [genre_id_to_name.get(g, str(g)) for g in genres]

else:
    movie_to_genre = {}

def get_genres(movie_title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": API_KEY, "query": movie_title}

    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()

        if data.get("results"):
            genre_ids = data["results"][0].get("genre_ids", [])
            return [genre_id_to_name.get(gid, str(gid)) for gid in genre_ids]
    except requests.RequestException as e:
        print(f"Request failed for '{movie_title}': {e}")

    return None

for i, movie in enumerate(movies, start=1):
    if movie in movie_to_genre:
        continue  

    genres = get_genres(movie)
    movie_to_genre[movie] = genres


    if i % 25 == 0 or i == total_movies:
        print(f"{i}/{total_movies} movies processed")
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(movie_to_genre, f, ensure_ascii=False, indent=2)

    time.sleep(REQUEST_DELAY)

print("Genre lookup complete")

df["genres"] = df["Movie"].map(movie_to_genre)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved dataset with genres to: {OUTPUT_CSV}")
