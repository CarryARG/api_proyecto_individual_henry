from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from sklearn.metrics.pairwise import linear_kernel
import logging
import numpy as np

app = FastAPI()

# Configurar el registro
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar el DataFrame procesado y los artefactos
try:
    df = pd.read_csv('movies_dataframe.csv', dtype={'release_date': 'str', 'budget': 'float64', 'revenue': 'float64'})
    logger.info("DataFrame cargado exitosamente.")
except Exception as e:
    logger.error(f"Error al cargar el DataFrame: {e}")

# Convertir release_date a datetime y crear nuevas columnas para mes y día de la semana
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['release_month'] = df['release_date'].dt.month
df['release_day'] = df['release_date'].dt.dayofweek
df['release_year'] = df['release_date'].dt.year

# Indexar la columna title para acelerar las búsquedas
df.set_index('title', inplace=True, drop=False)

# Cargar el vectorizador y la matriz TF-IDF
try:
    tfidf = joblib.load('tfidf_vectorizer.joblib')
    tfidf_matrix = joblib.load('tfidf_matrix.joblib')
    logger.info("Artefactos TF-IDF cargados exitosamente.")
except Exception as e:
    logger.error(f"Error al cargar los artefactos TF-IDF: {e}")

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
    try:
        if titulo not in df['title'].values:
            logger.error(f"Película no encontrada: {titulo}")
            return {"error": "Película no encontrada"}

        # Asegurarse de que idx sea un entero
        idx = df.index.get_loc(titulo)

        # Generar las similitudes
        cosine_sim = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
        print("cosine_sim:", cosine_sim)  # Verificar los valores y tipos de datos
        print("cosine_sim.dtype:", cosine_sim.dtype)  # Imprimir el tipo de dato

        # Manejar el caso de una única película
        if cosine_sim.size == 1:  # Comprobar el tamaño del array
            logger.info(f"No se encontraron suficientes películas para comparar con {titulo}.")
            return {"message": "No se encontraron suficientes películas similares"}

        # Comparación segura usando numpy.any() y asegurando tipos de datos
        threshold = 0.8  # Ajusta el umbral según sea necesario
        if np.any(cosine_sim.astype(float) > threshold):  # Convertir a float para asegurar comparación numérica
            # Obtener los índices de las películas similares
            similar_indices = np.where(cosine_sim > threshold)[0]
            # Obtener los títulos de las películas similares
            recommendations = df.iloc[similar_indices]['title'].tolist()
            logger.info(f"Recomendaciones para {titulo}: {recommendations}")
            return recommendations
        else:
            logger.info(f"No se encontraron recomendaciones para {titulo}")
            return {"message": "No se encontraron películas similares"}

    except IndexError:
        logger.error(f"Película no encontrada: {titulo}")
        return {"error": "Película no encontrada"}
    except ValueError as e:
        if "The truth value of an array with more than one element is ambiguous" in str(e):
            logger.error("Error en la comparación de arreglos. Revisa la lógica de comparación.")
            logger.error(f"Detalles del error: {e}")
            # Imprimir más detalles para depurar
            print("cosine_sim:", cosine_sim)
            print("threshold:", threshold)
            print("types:", type(cosine_sim), type(threshold))
        else:
            logger.error(f"Error inesperado: {e}")
        return {"error": "Ocurrió un error durante la recomendación"}

