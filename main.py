from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from sklearn.metrics.pairwise import linear_kernel
import logging

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
    raise HTTPException(status_code=500, detail="Error al cargar el DataFrame")

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
    raise HTTPException(status_code=500, detail="Error al cargar los artefactos TF-IDF")

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

@app.get("/recomendacion/{titulo}")
def recomendacion(titulo: str):
    try:
        # Verificar si el título está en el DataFrame
        if titulo.lower() not in df['title'].str.lower().values:
            logger.info(f"Título '{titulo}' no encontrado en la base de datos.")
            return {"error": "La película no se encuentra en la base de datos"}

        idx = df.index[df['title'].str.lower() == titulo.lower()].tolist()
        if not idx:
            logger.info(f"No se encontró el índice para el título '{titulo}'.")
            return {"error": "No se encontró el índice para el título"}

        idx = idx[0]

        # Calcular la similitud
        cosine_sim = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim[0]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # omitir la primera porque es la misma película

        movie_indices = [i[0] for i in sim_scores]
        recomendaciones  = df['title'].iloc[movie_indices].tolist()

        return recomendaciones

    except ValueError as e:
        logger.error(f"Error de valor en la recomendación: {e}")
        return {"error": "Ocurrió un error al calcular la similitud. Por favor, verifica los datos de entrada."}
    except IndexError as e:
        logger.error(f"Índice fuera de rango: {e}")
        return {"error": "No se encontraron películas similares. El título proporcionado podría estar mal escrito o no tener suficientes películas similares."}
    except Exception as e:
        logger.error(f"Error inesperado en la recomendación: {e}")
        return {"error": "Ocurrió un error interno. Por favor, intenta nuevamente más tarde."}
