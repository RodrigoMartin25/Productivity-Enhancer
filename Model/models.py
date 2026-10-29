"""Model definitions."""

import torch
import torch.nn as nn
from transformers import AutoModel


class ProdEnhancer_tapering(nn.Module):
    """DeBERTa with the original two-layer, GELU classification head."""
    def __init__(self, model_name="microsoft/deberta-v3-xsmall", num_labels=3, head_hidden_size=192, freeze_backbone=False):
        
        super().__init__()

        # Every instance gets its own pretrained backbone.
        # Checkpoints may load in float16 by default. Keep trainable weights and
        # Adam's optimizer state in float32 to avoid half-precision underflow.
        self.Deberta = AutoModel.from_pretrained(model_name, dtype=torch.float32)
        self.dense_1 = nn.Linear(self.Deberta.config.hidden_size, head_hidden_size)
        self.activation = nn.GELU()
        self.dense_2 = nn.Linear(head_hidden_size, num_labels)

        # Freeze DeBerta to only train the classification head
        self.freeze_backbone = freeze_backbone
        self.Deberta.requires_grad_(not freeze_backbone)
        if freeze_backbone:
            self.Deberta.eval()

    def train(self, mode=True):
        super().train(mode)

        # Disable backbone dropout when using it as a frozen feature extractor.
        if self.freeze_backbone:
            self.Deberta.eval()

        return self

    def forward(self, input_ids, attention_mask=None):
        outputs = self.Deberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :].float()
        outputs = self.dense_1(cls_embedding)
        outputs = self.activation(outputs)
        outputs = self.dense_2(outputs)

        return outputs


def build_model(config): # EXTEND BY DIFFERENT MODELS
    """Central place to register additional classification heads later."""
    if config["head"] != "tapering":
        raise ValueError(f"Unknown classification head: {config['head']}")
    return ProdEnhancer_tapering(
        model_name=config["model_name"],
        num_labels=config["num_labels"],
        head_hidden_size=config["head_hidden_size"],
        freeze_backbone=config["freeze_backbone"],
    )
