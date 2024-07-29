# Preproceso ML
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Cargar el dataset limpio y preprocesar
df = pd.read_parquet('../dataset_limpio.parquet')

# Handle missing titles (replace None with empty string)
df['title'] = df['title'].fillna('') 

# Convertir la Serie de pandas a una lista de strings
titulos = df['title'].tolist()

# Eliminar valores nulos explícitamente
titulos_limpios = [titulo for titulo in titulos if titulo is not None] 

# Limpieza adicional (puedes ajustar esto según tus necesidades)
titulos_limpios = [''.join(c for c in titulo if c.isalnum() or c.isspace()) 
                   for titulo in titulos_limpios if titulo]

# Crear el TfidfVectorizer basado en los títulos de las películas
tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

# Aplicar TfidfVectorizer a los títulos limpios
tfidf_matrix = tfidf.fit_transform(titulos_limpios)

# Guardar el TfidfVectorizer y la matriz TF-IDF
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
joblib.dump(tfidf_matrix, 'tfidf_matrix.joblib')
joblib.dump(df, 'movies_df.joblib')