from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import pickle
import ast 

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# --- LOAD MODEL ---
print("Loading StreamSense AI Model...") 
try:
    movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
    movies = pd.DataFrame(movies_dict)
    model = pickle.load(open('model.pkl', 'rb'))
    tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
    print("Model Loaded Successfully!") 
except FileNotFoundError:
    print("Error: Model files not found.")

def get_recommendations(movie_title):
    matches = movies[movies['original_title'].str.contains(movie_title, case=False, na=False)]
    
    if matches.empty:
        return None
    
    idx = matches.index[0]
    
    # KNN Search
    distances, indices = model.kneighbors(tfidf_matrix[idx], n_neighbors=7)
    movie_indices = indices[0][1:]
    
    results = []
    for i in movie_indices:
        # Parse Genres
        try:
            genres_list = [g['name'] for g in ast.literal_eval(movies['genres'].iloc[i])]
            genre_str = " • ".join(genres_list[:2])
        except:
            genre_str = "General"

        results.append({
            "title": movies['original_title'].iloc[i],
            "overview": str(movies['overview'].iloc[i])[:150] + "...",
            "genre": genre_str,
            # --- CHANGE: Using the REAL 'vote_average' column ---
            "rating": round(movies['vote_average'].iloc[i], 1)
        })
    return results

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request, 
        "movies": movies['original_title'].head(5000).values,
        "recommendations": [],
        "error": None
    })

@app.post("/recommend")
def recommend(request: Request, movie_name: str = Form(...)):
    results = get_recommendations(movie_name)
    
    error_msg = None
    if results is None:
        error_msg = f"Sorry, '{movie_name}' not found."
        results = []

    return templates.TemplateResponse("index.html", {
        "request": request, 
        "movies": movies['original_title'].head(5000).values, 
        "recommendations": results,
        "selected_movie": movie_name,
        "error": error_msg
    })