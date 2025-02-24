import argparse
from pathlib import Path
import pandas as pd
import torch
from functools import partial
from transformers import AutoModel, AutoTokenizer

from _models.huggingface.huggingface import get_device
from _models.model import get_embedding_func_batched
from make_dataset import data_path
from trainer import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train binary encoder")
    parser.add_argument("--local_model_names", type=str, nargs="+", default=["BAAI/bge-m3", "BAAI/bge-small-en", "google-bert/bert-base-uncased"], help="List of local model names")
    parser.add_argument("--api_model_names", type=str, nargs="+", default=[], help="List of API model names")
    parser.add_argument("--output_dir", type=str, default="project/models/binary_head", help="Output directory for models")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--temp", type=float, default=1.0, help="Temperature parameter")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--num_trainable_layers", type=int, default=2, help="Number of trainable layers in base model")
    
    return parser.parse_args()


def load_local_models(model_names, device):
    models = []
    tokenizers = []
    
    for model_name in model_names:
        print(f"Loading local model: {model_name}")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        models.append(model)
        tokenizers.append(tokenizer)
    
    return models, tokenizers


def prepare_api_embedding_funcs(model_names):
    embedding_funcs = []
    
    for model_name in model_names:
        print(f"Preparing API model: {model_name}")
        embedding_funcs.append(partial(get_embedding_func_batched(model_name)))
    
    return embedding_funcs


def prepare_data(test_ratio):
    print("Loading dataset...")
    data = pd.read_pickle(data_path)
    
    test_size = int(len(data) * test_ratio)
    test_data = data.sample(n=test_size, random_state=42)
    train_data = data.drop(test_data.index)
    
    print(f"Train set size: {len(train_data)}, Test set size: {len(test_data)}")
    return train_data, test_data


def main():
    args = parse_args()
    device = get_device(use_gpu=True)
    print(f"Using device: {device}")

    train_data, test_data = prepare_data(args.test_ratio)

    models, tokenizers = load_local_models(args.local_model_names, device)
    embedding_funcs = prepare_api_embedding_funcs(args.api_model_names)

    print("Starting training...")
    binary_head = train(
        models=models,
        tokenizers=tokenizers,
        embedding_funcs=embedding_funcs,
        train_data=train_data,
        test_data=test_data,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        temp=args.temp,
        num_trainable_layers=args.num_trainable_layers
    )

    binary_head.save_model(args.output_dir)


if __name__ == "__main__":
    main()