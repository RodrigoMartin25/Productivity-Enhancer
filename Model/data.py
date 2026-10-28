"""Dataset loading and batch tokenization shared by experiments."""

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class TitleCollector:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        encodings = self.tokenizer(
            [example["video_title"] for example in batch],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encodings["labels"] = torch.tensor(
            [example["productivity_label"] for example in batch],
            dtype=torch.long,
        )
        return encodings


def create_loaders(config):
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    collector = TitleCollector(tokenizer, config["max_length"])

    train_data = load_dataset(config["dataset_name"], split=config["train_split"])
    validation_data = load_dataset(config["dataset_name"], split=config["validation_split"])

    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_data,
        batch_size=config["batch_size"],
        shuffle=True,
        generator=generator,
        collate_fn=collector,
    )
    validation_loader = DataLoader(
        validation_data, 
        batch_size=config["batch_size"], 
        collate_fn=collector
    )
    return train_loader, validation_loader, tokenizer
