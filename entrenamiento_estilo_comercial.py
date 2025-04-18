import pandas as pd
import torch
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from datasets import Dataset

df = pd.read_csv("./entrenamiento_estilo_comercial.csv")
modelo_base = "./modelo_estilo_comercial"
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
    output_dir="./modelo_estilo_comercial",
    per_device_train_batch_size=4,
    num_train_epochs=60,
    learning_rate=3e-4,
    logging_dir="./logs_estilo",
    logging_steps=10,             
    report_to="tensorboard", 
    save_total_limit=1,
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
model.save_pretrained("./modelo_estilo_comercial")
tokenizer.save_pretrained("./modelo_estilo_comercial")
