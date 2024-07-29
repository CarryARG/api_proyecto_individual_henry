import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Cargar el dataset
df = pd.read_csv('dataset_limpio.csv')

# Vectorizar los títulos de las películas
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['title'])

# Guardar el vectorizador y la matriz TF-IDF
joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
joblib.dump(tfidf_matrix, 'tfidf_matrix.joblib')

# Guardar el DataFrame procesado
df.to_csv('movies_dataframe.csv', index=False)
