from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset (CSV from Kaggle)
dataset = load_dataset(
    "csv",
    data_files={
        "train": "train.csv",
        "test": "test.csv"
    }
)

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

def preprocess(batch):
    inputs = tokenizer(
        batch["article"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
    outputs = tokenizer(
        batch["highlights"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    inputs["labels"] = outputs["input_ids"]
    return inputs

dataset = dataset.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="./bart-summarizer",
    evaluation_strategy="steps",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=1,
    save_steps=1000,
    save_total_limit=2,
    logging_dir="./logs",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()
model.save_pretrained("trained_bart")
tokenizer.save_pretrained("trained_bart")
