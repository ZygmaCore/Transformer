import torch
import torch.nn as nn
from torch.nn.functional import dropout

from utilities import replikasi
from attention import MultiHeadAttention
from embedding import PositionalEncoding
from encoder import TransformerBlock

class DecoderBlock(nn.Module):
    def __init__(self, dimensi_embedding: int = 512, heads: int = 8, faktor_ekspansi: int = 4, dropout: float = 0.2) -> None:
        super(DecoderBlock, self).__init__()
        self.attention = MultiHeadAttention(dimensi_embedding, heads)
        self.norm = nn.LayerNorm(dimensi_embedding)
        self.dropout = nn.Dropout(dropout)
        self.transformerBlock = TransformerBlock(dimensi_embedding, heads, faktor_ekspansi, dropout)

    def forward(self, key: torch.Tensor, query: torch.Tensor, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        decoder_attention = self.attention(x, x, x mask)
        value = self.dropout(self.norm(decoder_attention + x))
        decoder_attention_output = self.transformerBlock(key, query, value)
        return decoder_attention_output

class Decoder(nn.Module):
    def __init__(self, ukuran_target_vocab: int, panjang_sekuens: int, dimensi_embedding: int = 512, jumlah_blocks: int = 6, faktor ekspansi: int = 4, heads: int = 8, dropout: float = 0.2) -> None:
        super(Decoder, self).__init__()
