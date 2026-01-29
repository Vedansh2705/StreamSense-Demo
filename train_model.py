import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

print("Starting StreamSense Training (20k + Real Ratings)...")

# 1. Load the Big Dataset
print("Loading dataset...")
try:
    df = pd.read_csv('movies_metadata.csv', low_memory=False)
except FileNotFoundError:
    print("Error: 'movies_metadata.csv' not found.")
    exit()

# 2. CLEAN & FILTER
# --- CHANGE: We added 'vote_average' to this list ---
df = df[['id', 'original_title', 'overview', 'genres', 'popularity', 'vote_average']]

df = df.dropna(subset=['overview', 'original_title'])

# Clean numeric columns
df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
df['vote_average'] = pd.to_numeric(df['vote_average'], errors='coerce').fillna(0)

# --- FILTER TOP 20,000 ---
print("Filtering top 20,000 movies by popularity...")
df = df.sort_values('popularity', ascending=False).head(20000)
df = df.reset_index(drop=True)

print(f"Training on {df.shape[0]} movies.")

# 3. VECTORIZATION
print("Vectorizing...")
tfidf = TfidfVectorizer(stop_words='english', max_features=20000)
tfidf_matrix = tfidf.fit_transform(df['overview'])

# 4. TRAIN MODEL
print("Training Nearest Neighbors Model...")
nn_model = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine')
nn_model.fit(tfidf_matrix)

# 5. SAVE FILES
print("Saving model files...")
pickle.dump(df.to_dict(), open('movies_dict.pkl', 'wb'))
pickle.dump(nn_model, open('model.pkl', 'wb'))
pickle.dump(tfidf_matrix, open('tfidf_matrix.pkl', 'wb'))

print("Success! Ratings updated.")