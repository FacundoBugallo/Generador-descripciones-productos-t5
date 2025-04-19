import pandas as pd
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from datasets import Dataset
import shutil
import unidecode

csv_path = "dataset_ventas_2000.csv"
df = pd.read_csv(csv_path)

for col in ["input_text", "target_text"]:
    df[col] = df[col].astype(str).apply(unidecode.unidecode)

modelo_base = "./modelo_t5_base"
tokenizer = T5Tokenizer.from_pretrained(modelo_base)
model = T5ForConditionalGeneration.from_pretrained(modelo_base)
dataset = Dataset.from_pandas(df)

def tokenize(example):
    return tokenizer(
        example["input_text"],
        text_target=example["target_text"],
        max_length=128,
        padding="max_length",
        truncation=True
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="./modelo_t5_base",
    per_device_train_batch_size=1,        
    gradient_accumulation_steps=2,             
    num_train_epochs=30,
    learning_rate=3e-4,
    logging_dir="./logs_t5_base",
    logging_steps=10,
    report_to="tensorboard",
    save_strategy="no",
    fp16=False,
    evaluation_strategy="no"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

trainer.train()

model.save_pretrained("./modelo_t5_base", safe_serialization=False)
tokenizer.save_pretrained("./modelo_t5_base")

print("\n🌟 Entrenamiento completado y modelo guardado en './modelo_t5_base'")
