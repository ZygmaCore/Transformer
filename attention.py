from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, dimensi_embedding: int = 512, heads: int = 8) -> None:
        super(MultiHeadAttention, self).__init__()
        self.dimensi_embedding = dimensi_embedding
        self.heads = heads
        self.head = int(self.dimensi_embedding / self.heads)

        self.query = nn.Linear(self.head, self.head, bias=False)
        self.key = nn.Linear(self.head, self.head, bias=False)
        self.value = nn.Linear(self.head, self.head, bias=False)

        self.fc_output = nn.Linear(self.head * self.heads, dimensi_embedding)