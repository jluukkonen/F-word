import os
import json
import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight

# Constants
MODEL_NAME = "cardiffnlp/twitter-roberta-base"
NUM_LABELS = 5
MAX_LENGTH = 256
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
SEED = 42

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    kappa = cohen_kappa_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "kappa": kappa
    }

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def train():
    # Paths
    train_path = "/Volumes/United/Work/F-word/swearing-nlp/data/processed/train.csv"
    test_path = "/Volumes/United/Work/F-word/swearing-nlp/data/processed/test.csv"
    output_dir = "/Volumes/United/Work/F-word/swearing-nlp/models/pilot_v1"
    log_path = "/Volumes/United/Work/F-word/swearing-nlp/results/training_log.json"
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Preprocess datasets
    def tokenize_function(examples):
        return tokenizer(examples["formatted_text"], truncation=True, padding=False, max_length=MAX_LENGTH)
    
    train_ds = Dataset.from_pandas(train_df[['formatted_text', 'label_id']])
    test_ds = Dataset.from_pandas(test_df[['formatted_text', 'label_id']])
    
    train_ds = train_ds.map(tokenize_function, batched=True).rename_column("label_id", "labels")
    test_ds = test_ds.map(tokenize_function, batched=True).rename_column("label_id", "labels")
    
    # Class weights
    labels = train_df['label_id'].values
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    print(f"Computed class weights: {class_weights}")
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        seed=SEED,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
        report_to="none"
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights
    )
    
    # Train
    print("Starting training...")
    train_result = trainer.train()
    
    # Save model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")
    
    # Save log
    with open(log_path, 'w') as f:
        json.dump(trainer.state.log_history, f, indent=4)
    print(f"Training log saved to {log_path}")

if __name__ == "__main__":
    train()
