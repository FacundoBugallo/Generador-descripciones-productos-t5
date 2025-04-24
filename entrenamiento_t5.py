# entrenamiento_t5_base.py

import pandas as pd
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
import unidecode

# Cargar y preparar el dataset
csv_path = "dataset_ventas_2000.csv"
df = pd.read_csv(csv_path)

for col in ["input_text", "target_text"]:
    df[col] = df[col].astype(str).apply(unidecode.unidecode)

# Separar datos para entrenamiento y validacion
df_train = df.sample(frac=0.9, random_state=42)
df_val = df.drop(df_train.index)

dataset = DatasetDict({
    "train": Dataset.from_pandas(df_train.reset_index(drop=True)),
    "validation": Dataset.from_pandas(df_val.reset_index(drop=True))
})

# Cargar modelo y tokenizer
modelo_base = "modelo_t5_base"
tokenizer = T5Tokenizer.from_pretrained(modelo_base)
model = T5ForConditionalGeneration.from_pretrained(modelo_base)

# Tokenizar
def tokenize(example):
    return tokenizer(
        example["input_text"],
        text_target=example["target_text"],
        max_length=128,
        padding="max_length",
        truncation=True
    )

tokenized_datasets = dataset.map(tokenize, batched=True)

# Configuracion de entrenamiento
training_args = Seq2SeqTrainingArguments(
    load_best_model_at_end=True,
    output_dir="./modelo_t5_base",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=3e-4,
    logging_dir="./logs_t5_base",
    logging_steps=5,
    report_to="tensorboard",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    save_total_limit=2,
    fp16=False
)

# Entrenador
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Entrenar
trainer.train()

# Guardar modelo y tokenizer
model.save_pretrained("./modelo_t5_base", safe_serialization=False)
tokenizer.save_pretrained("./modelo_t5_base")

print("\n\U0001F31F Entrenamiento completado con validación y early stopping.")