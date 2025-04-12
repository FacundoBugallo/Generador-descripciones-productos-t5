Este proyecto entrena un modelo `T5-small` usando la librería `transformers` de Hugging Face para generar descripciones naturales de productos a partir de sus atributos estructurados como texto.

## 📦 Objetivo
Entrenar un modelo que dados los atributos de un producto (como color, forma, material, estado, etc.), genere una descripción textual coherente y detallada.

## 🔁 Flujo del Proyecto

1. Cargar un dataset con productos y atributos (CSV)
2. Preparar las entradas (input_text) y salidas (target_text)
3. Entrenar un modelo T5 con `Trainer`
4. Guardar el modelo fine-tuned
5. Usar el modelo para generar descripciones a partir de nuevos productos

## 🧪 Ejemplo de entrada

```
describir: producto: cámara; color: negra; tamaño: mediano; forma: rectangular; material: plástico; estado: nuevo
```

## 📝 Ejemplo de salida generada

```
Una cámara negra de tamaño mediano, forma rectangular, fabricada en plástico y en estado nuevo.
```
```

## ⚙️ Requisitos

Instalar las dependencias:
```bash
pip install transformers datasets torch pandas tqdm sentencepiece
```

## 🚀 Uso

### 1. Entrenar el modelo:
```bash
python entrenamiento_t5.py
```

### 2. Generar descripciones para un ejemplo:
```bash
python generar_descripciones.py
```

### 3. Generar descripciones en lote desde un CSV:
```bash
python generar_descripciones_csv.py
```

## 🧠 Créditos
Proyecto creado como demostración de fine-tuning de LLMs con T5 para tareas de NLP aplicadas a productos.
