from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import linear_kernel

app = FastAPI()

# Carga del dataset y los modelos preprocesados
df = pd.read_csv('recursosApi/dataframe_procesado.csv')
tfidf = joblib.load('recursosApi/tfidf_vectorizer.joblib')
tfidf_matrix = joblib.load('recursosApi/tfidf_matrix.joblib')

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

@app.get('/cantidad_filmaciones_mes/{mes}')
def cantidad_filmaciones_mes(mes: str):
    meses = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4,
        'mayo': 5, 'junio': 6, 'julio': 7, 'agosto': 8,
        'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    mes_numero = meses.get(mes.lower())
    if mes_numero:
        count = df[df['release_month'] == mes_numero].shape[0]
        return {f"{count} cantidad de películas fueron estrenadas en el mes de {mes}"}
    else:
        return {"error": "Mes no válido"}

@app.get('/cantidad_filmaciones_dia/{dia}')
def cantidad_filmaciones_dia(dia: str):
    dias = {
        'lunes': 0, 'martes': 1, 'miércoles': 2, 'jueves': 3,
        'viernes': 4, 'sábado': 5, 'domingo': 6
    }
    dia_numero = dias.get(dia.lower())
    if dia_numero is not None:
        count = df[df['release_day'] == dia_numero].shape[0]
        return {f"{count} cantidad de películas fueron estrenadas en los días {dia}"}
    else:
        return {"error": "Día no válido"}

@app.get('/score_titulo/{titulo}')
def score_titulo(titulo: str):
    film = df[df['title'].str.lower() == titulo.lower()]
    if not film.empty:
        score = film.iloc[0]['popularity']
        year = film.iloc[0]['release_year']
        return {f"La película {titulo} fue estrenada en el año {year} con un score/popularidad de {score}"}
    else:
        return {"error": "Película no encontrada"}

@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo: str):
    film = df[df['title'].str.lower() == titulo.lower()]
    if not film.empty:
        votos = film.iloc[0]['vote_count']
        promedio_votos = film.iloc[0]['vote_average']
        return {
            "title": titulo,
            "votos": votos,
            "promedio_votos": promedio_votos
        }
    else:
        return {"error": "Película no encontrada"}

@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor: str):
    films = df[df['actors'].str.contains(nombre_actor, case=False, na=False)]
    if not films.empty:
        cantidad = films.shape[0]
        retorno_total = films['return'].sum()
        retorno_promedio = films['return'].mean()
        return {
            "actor": nombre_actor,
            "cantidad_peliculas": cantidad,
            "retorno_total": retorno_total,
            "retorno_promedio": retorno_promedio
        }
    else:
        return {"error": "Actor no encontrado"}

@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
    films = df[df['directors'].str.contains(nombre_director, case=False, na=False)]
    if not films.empty:
        peliculas_info = films[['title', 'release_date', 'return', 'budget', 'revenue']].to_dict(orient='records')
        return {
            "director": nombre_director,
            "peliculas": peliculas_info
        }
    else:
        return {"error": "Director no encontrado"}

@app.get('/recomendacion/{titulo}')
def recomendacion(titulo: str):
    # Verificamos si el título existe en el DataFrame
    if titulo not in df['title'].values:
        raise HTTPException(status_code=404, detail="Película no encontrada")

    # Obtenemos el índice de la película tipeada
    idx = df.index[df['title'] == titulo].tolist()[0]

    # Calculamos la similitud del coseno
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

    # Obtenemos los índices de las películas más similares
    similar_indices = cosine_sim.argsort()[-(6 + 1):][::-1]
    similar_indices = [i for i in similar_indices if i != idx]

    # Obtenemos los títulos de las películas "más similares"
    top_recommendations = []
    seen_titles = set()

    for index in similar_indices:
        title = df['title'].iloc[index]
        if title not in seen_titles:
            top_recommendations.append(title)
            seen_titles.add(title)
        if len(top_recommendations) == 5:
            break

    # Se aseguran que siempre haya 5 recomendaciones, llenando con títulos adicionales si es necesario
    if len(top_recommendations) < 5:
        additional_indices = cosine_sim.argsort()[-(6 + 5):][::-1]
        additional_indices = [i for i in additional_indices if i not in similar_indices and i != idx]
        for index in additional_indices:
            title = df['title'].iloc[index]
            if title not in seen_titles:
                top_recommendations.append(title)
                seen_titles.add(title)
            if len(top_recommendations) == 5:
                break

    return {"recommendations": top_recommendations}