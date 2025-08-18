# finetune_distilbert.py
import os
import argparse
import numpy as np
import pandas as pd
from datasets import Dataset, ClassLabel
from evaluate import load as load_metric
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.preprocessing import LabelEncoder

def load_dataframe(path, text_col='job_description', label_col='category'):
    df = pd.read_csv(path)
    # coalesce: if job_description missing use title/name
    df[text_col] = df[text_col].fillna('')  # adjust if you built raw_text earlier
    df = df[[text_col, label_col]].dropna(subset=[text_col])
    return df

def prepare_dataset(df, tokenizer, text_col='job_description', label_col='category'):
    le = LabelEncoder()
    labels = le.fit_transform(df[label_col].astype(str))
    df2 = df.copy()
    df2['label'] = labels
    ds = Dataset.from_pandas(df2[[text_col,'label']].rename(columns={text_col:'text'})).cast_column('label', ClassLabel(num_classes=len(le.classes_), names=list(le.classes_)))
    # tokenize
    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)
    ds = ds.map(tokenize, batched=True)
    ds.set_format(type="torch", columns=['input_ids', 'attention_mask', 'label'])
    ds = ds.shuffle(seed=42)
    # keep label mapping
    label_map = {i: c for i, c in enumerate(le.classes_)}
    return ds.train_test_split(test_size=0.1), le, label_map

def compute_metrics(p):
    metric = load_metric("f1")
    preds = np.argmax(p.predictions, axis=1)
    f1 = metric.compute(predictions=preds, references=p.label_ids, average='weighted')
    acc = (preds == p.label_ids).mean()
    return {"accuracy": acc, "f1_weighted": f1['f1']}

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    df = load_dataframe(args.data_path, text_col=args.text_col, label_col=args.label_col)
    dsets, le, label_map = prepare_dataset(df, tokenizer, text_col=args.text_col, label_col=args.label_col)
    num_labels = len(le.classes_)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16 if args.device == "gpu" else 8,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        fp16=False if args.device=="cpu" else True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dsets['train'],
        eval_dataset=dsets['test'],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()
    # save
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    # save label encoder mapping
    with open(os.path.join(args.out_dir, "label_map.json"), "w") as f:
        import json
        json.dump(label_map, f)
    print("Saved fine-tuned model to", args.out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="dataset/ghana_jobs_detailed.csv")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--out_dir", type=str, default="distilbert_finetuned")
    parser.add_argument("--text_col", type=str, default="job_description")
    parser.add_argument("--label_col", type=str, default="category")
    parser.add_argument("--device", type=str, choices=["cpu","gpu"], default="cpu")
    args = parser.parse_args()
    main(args)
