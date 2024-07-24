from fastapi import FastAPI
import pandas as pd
import ast

app = FastAPI()

# Cargar el dataset
movies_df = pd.read_csv('movies_dataset_desanidado.csv')

@app.get("/")
def read_root():
    return {"message": "API de películas"}

@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    meses = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4,
        "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
        "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    mes_numero = meses.get(mes.lower())
    if not mes_numero:
        return {"error": "Mes no válido"}
    
    cantidad = movies_df[pd.to_datetime(movies_df['release_date']).dt.month == mes_numero].shape[0]
    return {"mes": mes, "cantidad": cantidad}

@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    dias = {
        "lunes": 0, "martes": 1, "miércoles": 2, "jueves": 3,
        "viernes": 4, "sábado": 5, "domingo": 6
    }
    dia_numero = dias.get(dia.lower())
    if dia_numero is None:
        return {"error": "Día no válido"}
    
    cantidad = movies_df[pd.to_datetime(movies_df['release_date']).dt.dayofweek == dia_numero].shape[0]
    return {"dia": dia, "cantidad": cantidad}

@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str):
    pelicula = movies_df[movies_df['title'].str.lower() == titulo.lower()]
    if pelicula.empty:
        return {"error": "Título no encontrado"}
    
    resultado = pelicula.iloc[0]
    return {
        "titulo": resultado['title'],
        "año": resultado['release_year'],
        "score": resultado['vote_average']
    }

@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):
    pelicula = movies_df[movies_df['title'].str.lower() == titulo.lower()]
    if pelicula.empty:
        return {"error": "Título no encontrado"}
    
    resultado = pelicula.iloc[0]
    if resultado['vote_count'] < 2000:
        return {"mensaje": "La película no cumple con la condición de al menos 2000 valoraciones"}
    
    return {
        "titulo": resultado['title'],
        "cantidad_votos": resultado['vote_count'],
        "promedio_votos": resultado['vote_average']
    }

@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    actor_data = movies_df[movies_df['cast'].apply(lambda x: nombre_actor in ast.literal_eval(x))]
    if actor_data.empty:
        return {"error": "Actor no encontrado"}
    
    cantidad_peliculas = actor_data.shape[0]
    total_return = actor_data['return'].sum()
    promedio_return = actor_data['return'].mean()
    
    return {
        "actor": nombre_actor,
        "cantidad_peliculas": cantidad_peliculas,
        "total_return": total_return,
        "promedio_return": promedio_return
    }

@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str):
    director_data = movies_df[movies_df['crew'].apply(lambda x: any(d['job'] == 'Director' and d['name'] == nombre_director for d in ast.literal_eval(x)))]
    if director_data.empty:
        return {"error": "Director no encontrado"}
    
    peliculas = []
    for _, row in director_data.iterrows():
        peliculas.append({
            "titulo": row['title'],
            "fecha_lanzamiento": row['release_date'],
            "retorno_individual": row['return'],
            "costo": row['budget'],
            "ganancia": row['revenue']
        })
    
    return {
        "director": nombre_director,
        "peliculas": peliculas
    }

