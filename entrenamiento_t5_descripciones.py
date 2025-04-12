import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

df = pd.read_csv("dataset_productos_extendido.csv")

def crear_prompt(row):
    return (
        f"describir: producto: {row['producto']}; color: {row['color']}; tamaño: {row['tamaño']}; "
        f"forma: {row['forma']}; material: {row['material']}; estado: {row['estado']}"
    )

def descripcion_template(row):
    return (
        f"Una {row['producto']} de color {row['color']}, "
        f"tamaño {row['tamaño']}, con forma {row['forma']}, "
        f"fabricada en {row['material']} y en estado {row['estado']}."
    )

df["input_text"] = df.apply(crear_prompt, axis=1)
df["target_text"] = df.apply(descripcion_template, axis=1)

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

dataset = Dataset.from_pandas(df[["input_text", "target_text"]])

def tokenize(example):
    return tokenizer(
        example["input_text"],
        text_target=example["target_text"],
        max_length=128,
        padding="max_length",
        truncation=True
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

# Configuración de entrenamiento
training_args = Seq2SeqTrainingArguments(
    output_dir="./modelo_descripciones_t5",
    evaluation_strategy="no",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    learning_rate=3e-4,
    save_total_limit=1,
    fp16=False,
    logging_dir="./logs"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

trainer.train()
model.save_pretrained("./modelo_final")
tokenizer.save_pretrained("./modelo_final")
