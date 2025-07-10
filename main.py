from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Movie Popularity Prediction API")

# Загружаем модель
model = joblib.load("model.pkl")

# Описание входных данных
class Features(BaseModel):
    orginal_is_english: float
    has_popular_company: float
    has_popular_crew: float
    title_length: float
    overview_length: float
    budget_per_runtime: float
    has_tagline: float
    relese_year: float
    keyword_count: float
    has_budget: float
    budget: float
    Action: float
    Adventure: float
    Animation: float
    Comedy: float
    Crime: float
    Documentary: float
    Drama: float
    Family: float
    Fantasy: float
    Foreign: float
    History: float
    Music: float
    Mystery: float
    Romance: float
    Science_Fiction: float
    TV_Movie: float
    Thriller: float
    Western: float
    runtime: float

@app.post("/predict")
def predict(features: Features):
    input_data = np.array([[getattr(features, field) for field in features.__annotations__]])
    prediction = model.predict(input_data)
    return {"prediction": float(prediction[0])}
