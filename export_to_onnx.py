# export_to_onnx.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import os

def export_onnx(model_dir, onnx_path, max_length=128, opset=11, device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    # Create a dummy input
    dummy_text = "This is a sample job description for export."
    inputs = tokenizer(dummy_text, return_tensors="pt", padding='max_length', truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # ONNX export: dynamic axes for batch and sequence length
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"}
        },
        opset_version=opset,
        do_constant_folding=True,
        keep_initializers_as_inputs=False
    )
    print("Exported ONNX to:", onnx_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="distilbert_finetuned")
    parser.add_argument("--onnx_path", type=str, default="distilbert_finetuned/model.onnx")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.onnx_path), exist_ok=True)
    export_onnx(args.model_dir, args.onnx_path)
