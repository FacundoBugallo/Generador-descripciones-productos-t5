import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

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

if __name__ == "__main__":
    entrada = "describir: producto: cámara; color: negra; tamaño: mediano; forma: rectangular; material: plástico; estado: nuevo"
    descripcion = generar_descripcion(entrada)
    print("📥 Entrada:")
    print(entrada)
    print("\n📝 Descripción generada:")
    print(descripcion)
