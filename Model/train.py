"""Run one independent experiment: python -m Model.train --config PATH."""

import argparse
import json
import random
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import create_loaders
from .evaluate import evaluate
from .models import build_model


def train(config):
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    # A unique directory prevents accidentally overwriting another experiment.
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    if config["validation_split"] == "test":
        warnings.warn(
            "The test split is being used for validation, as in the original script. ",
            stacklevel=2,
        )

    device = torch.device(config.get("device", "cpu"))

    train_loader, validation_loader, tokenizer = create_loaders(config)
    model = build_model(config).to(device)
    tokenizer.save_pretrained(output_dir)

    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["learning_rate"],
    )

    history = []
    best_f1 = -1.0
    class_counts = torch.tensor([85, 252, 530])
    weights = class_counts.sum() / (len(class_counts) * class_counts)

    # Training Loop
    for epoch in range(1, config["epochs"] + 1):

        model.train()

        total_loss = 0.0
        count = 0

        for batch in train_loader:

            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )

            loss = F.cross_entropy(logits, labels, weight=weights)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.numel()
            count += labels.numel()

        
        metrics = {
            "epoch": epoch,
            "train_loss": total_loss / count,
            "validation": evaluate(model, validation_loader, device, config["num_labels"]),
        }

        # Summary of metrics on each Epoch
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        (output_dir / "metrics.json").write_text(json.dumps(history, indent=2) + "\n")
        if metrics["validation"]["macro_f1"] > best_f1:
            best_f1 = metrics["validation"]["macro_f1"]
            torch.save(
                {"model_state_dict": model.state_dict(), "config": config, "epoch": epoch},
                output_dir / "best.pt",
            )

    return history

# take the config file path and load content as config variable
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["epochs"] < 1:
        parser.error("epochs must be at least 1")
    train(config)


if __name__ == "__main__":
    main()
