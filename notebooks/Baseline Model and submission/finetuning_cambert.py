import pandas as pd

path = "train_clean.jsonl"
path_test = "test_clean.jsonl"

df = pd.read_json(path, lines=True)
df_test = pd.read_json(path_test, lines=True)

import pandas as pd
import numpy as np

LABEL_COL = "label"

TEXT_COLS = ["text_clean", "user.description", "user.location"]

# Make sure they exist
print([c for c in TEXT_COLS if c not in df.columns])  # should print [] ideally

# Build one combined text column
df["text_all"] = (
    df["text_clean"].fillna("") + " [DESC] " +
    df["user.description"].fillna("") + " [LOC] " +
    df["user.location"].fillna("")
).str.strip()

texts = df["text_all"].astype(str).values
y = df[LABEL_COL].astype(int).values

from sklearn.model_selection import train_test_split

texts_train, texts_val, y_train, y_val = train_test_split(
    texts,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
# If needed:
# !pip install transformers accelerate -q

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_name = "camembert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=256,          # tweets + desc + loc → 128 is fine
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_ds = TweetDataset(texts_train, y_train)
val_ds   = TweetDataset(texts_val,   y_val)
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,   # binary classification
)
model.to(device)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "auc": auc}
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="/Data/camembertinfluencer_e",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=7,
    weight_decay=0.01,
    logging_steps=100,
    do_eval=True,   # old versions use this flag
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)
trainer.train()
eval_res = trainer.evaluate()
print(eval_res)

from pathlib import Path
import json

OUTPUT_DIR = "/Data/camembertinfluencer_e"  # or any scratch dir you like
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

trainer.save_model(OUTPUT_DIR)      # saves model + config
tokenizer.save_pretrained(OUTPUT_DIR)

with open(f"{OUTPUT_DIR}/eval_results.json", "w") as f:
        json.dump({'eval_loss': 0.2962910532951355,
                           'eval_accuracy': 0.900074234257496,
                                          'eval_auc': 0.9555951652607091}, f, indent=2)

