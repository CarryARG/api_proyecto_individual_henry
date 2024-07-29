from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Cargar el dataset y modelos preentrenados
df = joblib.load('ML/movies_df.joblib')
tfidf = joblib.load('ML/tfidf_vectorizer.joblib')
tfidf_matrix = joblib.load('ML/tfidf_matrix.joblib')

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
            "/recomendacion/{titulo}": "Devuelve una lista de 5 películas similares al título especificado."
        },
    }

# Convertir release_date a datetime y crear nuevas columnas para mes y día de la semana
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_month'] = df['release_date'].dt.month
df['release_day'] = df['release_date'].dt.dayofweek

# Indexar la columna title para acelerar las búsquedas
df.set_index('title', inplace=True, drop=False)

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
            "/dataset_info?page={pagina}&page_size=10": "Endpoint de prueba para revisar el dataset desde el 0 hasta el 453, con un tamaño de 10"
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
    if titulo in df.index:
        pelicula = df.loc[titulo]
        return {
            'titulo': pelicula['title'],
            'año': pelicula['release_year'],
            'score': pelicula['vote_average']
        }
    else:
        return {"error": "Título no encontrado"}

@app.get('/votos_titulo/{titulo}')
def votos_titulo(titulo: str):
    if titulo in df.index:
        pelicula = df.loc[titulo]
        if pelicula['vote_count'] >= 2000:
            return {
                'titulo': pelicula['title'],
                'cantidad_votos': pelicula['vote_count'],
                'promedio_votos': pelicula['vote_average']
            }
        else:
            return {"error": "La película no cumple con los votos suficientes"}
    else:
        return {"error": "Título no encontrado"}

@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor: str):
    actor_peliculas = df[df['actors'].str.contains(nombre_actor, na=False)]
    if not actor_peliculas.empty:
        cantidad_peliculas = actor_peliculas.shape[0]
        retorno_promedio = actor_peliculas['return'].mean()
        exito = actor_peliculas['return'].sum()
        return {
            'actor': nombre_actor,
            'cantidad_peliculas': cantidad_peliculas,
            'retorno_promedio': retorno_promedio,
            'exito': exito
        }
    else:
        return {"error": "Actor no encontrado"}

@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
    director_peliculas = df[df['directors'].str.contains(nombre_director, na=False)]
    if not director_peliculas.empty:
        peliculas = director_peliculas[['title', 'release_date', 'return', 'budget', 'revenue']].to_dict(orient='records')
        exito = director_peliculas['return'].sum()
        return {
            'director': nombre_director,
            'peliculas': peliculas,
            'exito': exito
        }
    else:
        return {"error": "Director no encontrado"}


@app.get('/recomendacion/{titulo}', name="Sistema de recomendación")
def recomendacion(titulo: str):
    # Obtener el índice de la película
    idx = df[df['title'] == titulo].index
    if len(idx) == 0:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    idx = idx[0]
    
    # Verifica que el índice esté dentro del rango
    if idx < 0 or idx >= tfidf_matrix.shape[0]:
        raise HTTPException(status_code=500, detail="Índice fuera de rango")
    
    # Calcular similitudes
    try:
        cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Obtener los índices de las películas más similares
    indices_similares = cosine_sim.argsort()[-6:-1]
    peliculas_similares = df['title'].iloc[indices_similares].tolist()
    
    return {"peliculas_similares": peliculas_similares}

# Ejemplo para correr la aplicación: `uvicorn main:app --reload`