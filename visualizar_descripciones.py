# visualizar_descripciones.py

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

# 1. Cargar CSV generado por el modelo (editá si el nombre cambia)
df = pd.read_csv("descripciones_generadas.csv")

# 2. Estadísticas básicas
df["longitud"] = df["descripcion_generada"].str.split().str.len()

print("\nResumen de longitudes de descripciones:")
print(df["longitud"].describe())

# 3. Histograma de longitud
plt.figure(figsize=(10, 5))
plt.hist(df["longitud"], bins=10, color="skyblue", edgecolor="black")
plt.title("Distribución de longitud de descripciones generadas")
plt.xlabel("Cantidad de palabras")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Nube de palabras
texto_completo = " ".join(df["descripcion_generada"].dropna().astype(str))
word_freq = Counter(texto_completo.lower().split())

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Nube de palabras en descripciones generadas")
plt.tight_layout()
plt.show()

# 5. Guardar CSV con longitud por si querés analizar en Excel
df.to_csv("descripciones_generadas_con_longitud.csv", index=False)
print("\nArchivo actualizado: descripciones_generadas_con_longitud.csv")
