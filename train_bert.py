import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import joblib

# Load data
df = pd.read_csv("blooms_taxonomy_dataset.csv")

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["Category"])
joblib.dump(le, "label_encoder.pkl")  # Save encoder

# Tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# Tokenize function
def tokenize(example):
    return tokenizer(example["Questions"], truncation=True, padding='max_length', max_length=128)

# Split manually using sklearn
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["Questions"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# Convert to HuggingFace datasets
train_dataset = Dataset.from_dict({"Questions": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"Questions": test_texts, "label": test_labels})

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

# Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(le.classes_))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="no"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Train
trainer.train()

# Save model
model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")

# Evaluate
preds = trainer.predict(test_dataset)
y_pred = np.argmax(preds.predictions, axis=1)
y_true = preds.label_ids

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=le.classes_))
