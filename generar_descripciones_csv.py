
import torch
import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm

modelo_path = "./modelo_final"
tokenizer = T5Tokenizer.from_pretrained(modelo_path)
model = T5ForConditionalGeneration.from_pretrained(modelo_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
def generar_descripcion(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=64,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

df = pd.read_csv("dataset_productos_extendido.csv")

if "input_text" not in df.columns:
    def crear_prompt(row):
        return (
            f"describir: producto: {row['producto']}; color: {row['color']}; tamaño: {row['tamaño']}; "
            f"forma: {row['forma']}; material: {row['material']}; estado: {row['estado']}"
        )
    df["input_text"] = df.apply(crear_prompt, axis=1)

tqdm.pandas()
df["descripcion_generada"] = df["input_text"].progress_apply(generar_descripcion)

df.to_csv("dataset_con_descripciones_generadas.csv", index=False)
print("✅ Descripciones generadas y guardadas en 'dataset_con_descripciones_generadas.csv'")