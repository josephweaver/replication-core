import torch
from torch import nn


class DiagonalGaussianEmbedding(nn.Module):
    """
    Stores one diagonal Gaussian embedding per item.

    Each item has:
    - mean vector mu
    - log-variance vector logvar
    """

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()

        self.mu = nn.Embedding(num_embeddings, embedding_dim)
        self.logvar = nn.Embedding(num_embeddings, embedding_dim)

        nn.init.normal_(self.mu.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.logvar.weight, 0.0)

    def forward(self, indices):
        return self.mu(indices), self.logvar(indices)