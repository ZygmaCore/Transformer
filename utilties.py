import copy
import torch.nn as nn

def replikasi(block, N: int = 6):
    block_stack = nn.ModuleList([copy.deepcopy(block) for _ in range(N)])