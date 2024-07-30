from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from wordcloud import WordCloud

import matplotlib.pyplot as plt
import re

app = FastAPI()

# Cargar el dataset limpio y preprocesar
df = pd.read_csv('dataset_limpio.csv')

# Convertir release_date a datetime y crear nuevas columnas para mes y día de la semana
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_month'] = df['release_date'].dt.month
df['release_day'] = df['release_date'].dt.dayofweek
df['release_year'] = df['release_date'].dt.year

# Indexar la columna title para acelerar las búsquedas
df.set_index('title', inplace=True, drop=False)

# Asegurarse que la columna 'title' sea de tipo string y limpiarla de caracteres no alfanuméricos
df['title'] = df['title'].astype(str)
df['title'] = df['title'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x))

# Generar una nube de palabras a partir de los títulos de las películas
try:
    text = ' '.join(df['title'].values)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('wordcloud.png')   
  
  # Guardar la nube de palabras como una imagen
    plt.close()
except TypeError as e:
    print(f"Error al generar la nube de palabras: {e}")
    # Identificar y manejar los títulos problemáticos
    for index, row in df.iterrows():
        if not isinstance(row['title'], str):
            print(f"Row {index} has a non-string value in the 'title' column: {row['title']}")

# Vectorizar los títulos de las películas
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['title'])

# Guardar el vectorizador y la matriz TF-IDF
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
joblib.dump(tfidf_matrix, 'tfidf_matrix.joblib')

# Guardar el DataFrame procesado
df.to_csv('movies_dataframe.csv', index=False)

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
            "/recomendacion/{titulo}": "Devuelve una lista de 5 películas similares al título especificado.",
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
    if titulo not in df['title'].values:
        return {"error": "Película no encontrada"}

    idx = df[df['title'] == titulo].index[0]
    cosine_sim = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Excluir la propia película
    movie_indices = [i[0] for i in sim_scores]
    
    return [df['title'].iloc[i] for i in movie_indices]
