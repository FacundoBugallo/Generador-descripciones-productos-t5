import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

df = pd.read_csv("descripciones_generadas.csv")
df["longitud"] = df["descripcion_generada"].str.split().str.len()

print("\nResumen de longitudes de descripciones:")
print(df["longitud"].describe())

plt.figure(figsize=(10, 5))
plt.hist(df["longitud"], bins=10, color="skyblue", edgecolor="black")
plt.title("Distribución de longitud de descripciones generadas")
plt.xlabel("Cantidad de palabras")
plt.ylabel("Frecuencia")
plt.grid(True)
plt.tight_layout()
plt.show()

texto_completo = " ".join(df["descripcion_generada"].dropna().astype(str))
word_freq = Counter(texto_completo.lower().split())

wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Nube de palabras en descripciones generadas")
plt.tight_layout()
plt.show()

df.to_csv("descripciones_generadas_con_longitud.csv", index=False)
print("\nArchivo actualizado: descripciones_generadas_con_longitud.csv")
