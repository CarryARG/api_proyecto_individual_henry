import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv('dataset_limpio.csv')

# Generar una nube de palabras a partir de los títulos de las películas
text = ' '.join(df['title'].values)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

# Mostrar y guardar la nube de palabras
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('wordcloud.png')  # Guardar la nube de palabras como una imagen
plt.close()
