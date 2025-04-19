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

shutil.rmtree("./modelo_t5_base", ignore_errors=True)
shutil.rmtree("./logs_t5_base", ignore_errors=True)

csv_path = "entrenamiento_estilo_comercial.csv"
df = pd.read_csv(csv_path)

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")
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
    per_device_train_batch_size=4,
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
