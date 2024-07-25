from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import ast

app = FastAPI()

# Especificar tipos de datos para las columnas
dtype_dict = {
    'belongs_to_collection': str,
    'budget': float,
    'genres': str,
    'id': int,
    'original_language': str,
    'original_title': str,
    'overview': str,
    'popularity': float,
    'production_companies': str,
    'release_date': str,
    'revenue': float,
    'runtime': float,
    'spoken_languages': str,
    'status': str,
    'tagline': str,
    'title': str,
    'vote_average': float,
    'vote_count': int,
    'release_year': int,
    'return': float
}

df = pd.read_csv('movies_dataset_limpio.csv', dtype=dtype_dict)

# Cargar el dataset
#df = pd.read_csv('movies_dataset_limpio.csv', low_memory=False)

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
            "/dataset_info/": "Endpoint de prueba para revisar el dataset"
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
        count = df[pd.to_datetime(df['release_date'], errors='coerce').dt.month == mes_numero].shape[0]
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
        count = df[pd.to_datetime(df['release_date'], errors='coerce').dt.dayofweek == dia_numero].shape[0]
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

# Esto es opcional, es para revisar el dataset
@app.get("/dataset_info")
def dataset_info():
    try:
        # Reemplazar NaN y valores infinitos con None para que sean JSON serializables
        df_clean = df.replace({np.nan: None, np.inf: None, -np.inf: None})
        return {"columns": df_clean.columns.tolist(), "sample_data": df_clean.head().to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
