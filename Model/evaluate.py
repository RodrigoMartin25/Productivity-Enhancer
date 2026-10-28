"""Shared classification evaluation, without dataset or model downloads."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate(model, data_loader, device, num_labels=3):
    """Return example-averaged loss and metrics; matrix rows are true labels."""
    was_training = model.training
    model.eval()

    confusion = torch.zeros(num_labels, num_labels, dtype=torch.long)
    total_loss = 0.0
    count = 0

    try:
        for batch in data_loader:

            labels = batch["labels"].to(device)
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )

            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            predictions = logits.argmax(dim=-1)

            # Turns each (predicted vs correct) entry into a number so .bincount() can count how many of each there are more efficiently 
            indices = (labels * num_labels + predictions).cpu() 
            confusion += torch.bincount(
                indices, minlength=num_labels * num_labels
            ).reshape(num_labels, num_labels)

            count += labels.numel()
    finally:
        model.train(was_training)

    true_positives = confusion.diag().float()
    precision = true_positives / confusion.sum(dim=0).clamp(min=1)
    recall = true_positives / confusion.sum(dim=1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)

    return {
        "loss": total_loss / count,
        "accuracy": true_positives.sum().item() / count,
        "macro_f1": f1.mean().item(),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "confusion_matrix": confusion.tolist(),
    }
