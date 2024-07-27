from fastapi import FastAPI, HTTPException, Query
import pandas as pd
import numpy as np
import ast

app = FastAPI()

# Cargar el dataset limpio
df = pd.read_csv('dataset_limpio.csv')

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

@app.get("/dataset_info")
def dataset_info(skip: int = Query(0, alias="page", ge=0), limit: int = Query(10, le=100)):
    """
    Devuelve un subconjunto del dataset.
    
    - skip: número de la página para saltar (por defecto 0)
    - limit: número de registros por página (por defecto 10, máximo 100)
    """
    try:
        # Reemplazar NaN y valores infinitos con None para que sean JSON serializables
        df_clean = df.replace({np.nan: None, np.inf: None, -np.inf: None})
        
        # Seleccionar la página de datos
        start = skip * limit
        end = start + limit
        
        # Asegurarse de que no se superen los límites del DataFrame
        if start >= len(df_clean):
            raise HTTPException(status_code=404, detail="No hay más datos para mostrar.")
        
        # Extraer el subconjunto de datos
        subset = df_clean.iloc[start:end]
        
        return {"columns": df_clean.columns.tolist(), "data": subset.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
