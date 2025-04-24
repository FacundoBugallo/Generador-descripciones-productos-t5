# Generador de Descripciones Comerciales con T5

Este proyecto utiliza un modelo basado en T5 para generar descripciones comerciales de productos a partir de sus características. Está diseñado para simular un vendedor profesional que crea textos atractivos y persuasivos.

---

## 🚀 Características principales
- Entrenamiento supervisado con un dataset de 2000 ejemplos.
- Generación automática de descripciones desde línea de comandos.
- Tokenización y preprocesamiento personalizados.
- División del dataset en entrenamiento y validación.
- Uso de `EarlyStopping` y estrategia de evaluación por épocas.

---

## 🧠 Arquitectura del modelo
- **Modelo base:** T5 ("t5-base" o un modelo entrenado personalizado).
- **Entrenamiento:** PyTorch + Hugging Face Transformers.
- **Entradas:** texto tipo prompt con atributos del producto.
- **Salida esperada:** descripción comercial del producto.

---

## 🛠️ Cómo entrenar el modelo
1. Coloca tu dataset en `dataset_ventas_2000.csv` con columnas:
   - `input_text`: descripción estructurada del producto.
   - `target_text`: descripción estilo comercial.

2. Corre el script de entrenamiento:
```bash
python entrenamiento_t5_base.py
```

3. El modelo se guarda en:
```
./modelo_t5_base/
```

---

## 🧪 Cómo generar descripciones
Ejecuta el script `generar_comercial.py` para probar el modelo con ejemplos nuevos.

```bash
python generar_comercial.py
```

Este script usa el modelo guardado y genera 10 descripciones automáticas.

---

## 📊 Visualización
Puedes usar TensorBoard para monitorear el entrenamiento:
```bash
tensorboard --logdir=./logs_t5_base
```

---

## 📂 Estructura del proyecto
```
├── dataset_ventas_2000.csv         # Dataset de entrenamiento
├── modelo_t5_base/                 # Modelo entrenado
├── logs_t5_base/                   # Logs de entrenamiento para TensorBoard
├── entrenamiento_t5_base.py       # Script de entrenamiento
├── generar_comercial.py           # Script para generar descripciones
├── README.md                       # Este archivo
```

---

## 🧾 Requisitos
```bash
pip install transformers datasets torch pandas unidecode tensorboard
```

---

## 📌 Notas
- Asegúrate de no sobreentrenar: usa `early_stopping`.
- Prompts más detallados ayudan a mejores resultados.
- Revisa diversidad en tu dataset para evitar repeticiones.

---

## 🧠 Ejemplo de prompt:
```text
Producto: Celular
Color: Negro
Tamaño: 8 pulgadas
Forma: Rectangular
Material: Metálico
Estado: Nuevo
```

Genera algo como:
```text
Un moderno celular negro de 8 pulgadas con diseño rectangular metálico. Ideal para quienes buscan potencia y estilo en un solo dispositivo.
```

---

**Autor:** Facundo Bugallo

---
