import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import time

# Función para obtener datos de la API en lotes pequeños con reintentos
def obtener_datos_api(url_base, page_size=1000, max_retries=5):
    page = 0
    all_data = []
    
    while True:
        url = f"{url_base}?page={page}&page_size={page_size}"
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=10)  # Agregar tiempo de espera
                response.raise_for_status()  # Verificar si la solicitud fue exitosa
                data = response.json()
                
                if data['data']:
                    all_data.extend(data['data'])
                    page += 1
                    break
                else:
                    return pd.DataFrame(all_data)  # No más datos disponibles, retornar DataFrame
            except requests.exceptions.RequestException as e:
                print(f"Error al obtener los datos de la API en el intento {attempt + 1}: {e}")
                time.sleep(2)  # Esperar antes de reintentar
                
                if attempt == max_retries - 1:
                    print("Máximo número de reintentos alcanzado. Terminando proceso.")
                    return pd.DataFrame(all_data)
    
    return pd.DataFrame(all_data)

# URL base de la API
url_base = "https://api-proyecto-individual-henry-4.onrender.com/dataset_info"

# Obtener datos de la API
df = obtener_datos_api(url_base, page_size=0, max_retries=5)  # Cambiar page_size a 50 y agregar reintentos

if df.empty:
    print("No se pudieron obtener datos de la API, o ya finalizo la obtencion de datos.")
else:
    # Asegurarse de que las columnas 'popularity' y 'revenue' sean numéricas
    df['popularity'] = pd.to_numeric(df['popularity'], errors='coerce')
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')

    # 1. Nube de palabras de los títulos de las películas
    all_titles = ' '.join(df['title'].dropna().values)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_titles)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nube de Palabras de Títulos de Películas')
    plt.show()

    # 2. Gráfico de barras de las 10 películas con mayor recaudación
    top_10_revenue = df.nlargest(10, 'revenue')
    plt.figure(figsize=(10, 5))
    plt.barh(top_10_revenue['title'], top_10_revenue['revenue'], color='skyblue')
    plt.xlabel('Recaudación')
    plt.title('Top 10 Películas con Mayor Recaudación')
    plt.gca().invert_yaxis()
    plt.show()

    # 3. Gráfico de barras de las 10 películas con mejor popularidad
    top_10_popularity = df.nlargest(10, 'popularity')
    plt.figure(figsize=(10, 5))
    plt.barh(top_10_popularity['title'], top_10_popularity['popularity'], color='orange')
    plt.xlabel('Popularidad')
    plt.title('Top 10 Películas con Mejor Popularidad')
    plt.gca().invert_yaxis()
    plt.show()

    # 4. Gráfico de barras de la cantidad de filmaciones por día de la semana
    df['release_day'] = pd.to_datetime(df['release_date'], errors='coerce').dt.day_name()
    filmaciones_por_dia = df['release_day'].value_counts().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
    plt.figure(figsize=(10, 5))
    filmaciones_por_dia.plot(kind='bar', color='green')
    plt.xlabel('Día de la Semana')
    plt.ylabel('Cantidad de Filmaciones')
    plt.title('Cantidad de Filmaciones por Día de la Semana')
    plt.xticks(rotation=45)
    plt.show()

    # 5. Gráfico de barras de la cantidad de filmaciones por mes
    df['release_month'] = pd.to_datetime(df['release_date'], errors='coerce').dt.month_name()
    filmaciones_por_mes = df['release_month'].value_counts().reindex([
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ])
    plt.figure(figsize=(10, 5))
    filmaciones_por_mes.plot(kind='bar', color='purple')
    plt.xlabel('Mes')
    plt.ylabel('Cantidad de Filmaciones')
    plt.title('Cantidad de Filmaciones por Mes')
    plt.xticks(rotation=45)
    plt.show()  