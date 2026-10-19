from transformers import AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall", device_map="auto")
print(model)

class ProdEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = model
        self.layer_2 = model

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        return x 