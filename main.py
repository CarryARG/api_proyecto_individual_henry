from fastapi import FastAPI
import pandas as pd
import ast
from fastapi.responses import HTMLResponse

app = FastAPI()

# Cargar el dataset
df = pd.read_csv('movies_dataset_desanidado.csv')

@app.get("/", response_class=HTMLResponse)
def read_root():
    html_content = """
    <html>
        <head>
            <title>API de Películas</title>
        </head>
        <body>
            <h1>Bienvenido a la API de Películas</h1>
            <p>Utiliza los siguientes endpoints para obtener información:</p>
            <ul>
                <li>/cantidad_filmaciones_mes/{mes}</li>
                <li>/cantidad_filmaciones_dia/{dia}</li>
                <li>/score_titulo/{titulo}</li>
                <li>/votos_titulo/{titulo}</li>
                <li>/get_actor/{nombre_actor}</li>
                <li>/get_director/{nombre_director}</li>
            </ul>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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
        votes = film.iloc[0]['vote_count']
        average_vote = film.iloc[0]['vote_average']
        if votes >= 2000:
            year = film.iloc[0]['release_year']
            return {f"La película {titulo} fue estrenada en el año {year}. La misma cuenta con un total de {votes} valoraciones, con un promedio de {average_vote}"}
        else:
            return {"error": "La película no tiene al menos 2000 valoraciones"}
    else:
        return {"error": "Película no encontrada"}

@app.get('/get_actor/{nombre_actor}')
def get_actor(nombre_actor: str):
    actor_films = df[df['cast'].apply(lambda x: nombre_actor.lower() in [d['name'].lower() for d in ast.literal_eval(x) if 'name' in d])]
    if not actor_films.empty:
        total_return = actor_films['return'].sum()
        num_films = actor_films.shape[0]
        avg_return = total_return / num_films if num_films > 0 else 0
        return {f"El actor {nombre_actor} ha participado de {num_films} cantidad de filmaciones, el mismo ha conseguido un retorno de {total_return} con un promedio de {avg_return} por filmación"}
    else:
        return {"error": "Actor no encontrado"}

@app.get('/get_director/{nombre_director}')
def get_director(nombre_director: str):
    director_films = df[df['crew'].apply(lambda x: nombre_director.lower() in [d['name'].lower() for d in ast.literal_eval(x) if d['job'] == 'Director'])]
    if not director_films.empty:
        films_data = []
        for _, row in director_films.iterrows():
            films_data.append({
                "title": row['title'],
                "release_date": row['release_date'],
                "return": row['return'],
                "budget": row['budget'],
                "revenue": row['revenue']
            })
        return {f"El director {nombre_director} ha dirigido las siguientes películas": films_data}
    else:
        return {"error": "Director no encontrado"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
