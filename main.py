from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI()

# Cargar el dataset y los modelos preprocesados
df = pd.read_csv('dataframe_procesado.csv')
tfidf = joblib.load('tfidf_vectorizer.joblib')
tfidf_matrix = joblib.load('tfidf_matrix.joblib')

@app.get("/")
def read_root():
    return {
        "message": "Bienvenido a la API de Películas. Utiliza los siguientes endpoints para obtener información:",
        "endpoints": {
            "/cantidad_filmaciones_mes/{mes}": "Devuelve la cantidad de películas estrenadas en el mes especificado.",
            "/cantidad_filmaciones_dia/{dia}": "Devuelve la cantidad de películas estrenadas en el día especificado.",
            "/score_titulo/{titulo}": "Devuelve el título, año de estreno y score de la película especificada.",
            "/votos_titulo/{titulo}": "Devuelve el título, cantidad de votos y promedio de votaciones de la película especificada.",
            "/get_actor/{nombre_actor}": "Devuelve el éxito del actor especificado, cantidad de películas y promedio de retorno.",
            "/get_director/{nombre_director}": "Devuelve el éxito del director especificado, nombre de cada película, fecha de lanzamiento, retorno individual, costo y ganancia.",
            "/recomendacion/{titulo}": "Devuelve una lista de 5 películas similares basadas en el título especificado.",
        },
    }

@app.get('/recomendacion/{titulo}')
def recomendacion(titulo: str):
    # Verificar si el título existe en el DataFrame
    if titulo not in df['title'].values:
        raise HTTPException(status_code=404, detail="Película no encontrada")

    # Obtener el índice de la película
    idx = df[df['title'] == titulo].index[0]

    # Calcular la similitud del coseno
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Obtener los índices de las películas más similares
    similar_indices = cosine_sim.argsort()[::-1]

    # Verificar que los índices sean válidos y dentro del rango del DataFrame
    valid_indices = [i for i in similar_indices[1:6] if 0 <= i < len(df)]

    # Obtener los títulos de las películas más similares
    top_recommendations = [df['title'].iloc[i] for i in valid_indices]

    return {"recommendations": top_recommendations}
