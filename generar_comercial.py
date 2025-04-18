import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

modelo_path = "./modelo_estilo_comercial"
tokenizer = T5Tokenizer.from_pretrained(modelo_path)
model = T5ForConditionalGeneration.from_pretrained(modelo_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generar_descripcion(prompt="Generar una descripción comercial"):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=128,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        no_repeat_ngram_size=3
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== Descripciones generadas ===\n")
for i in range(10):
    descripcion = generar_descripcion()
    print(f"{i+1}. {descripcion}")