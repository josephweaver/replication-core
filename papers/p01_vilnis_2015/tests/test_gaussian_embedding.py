import torch

from papers.p01_vilnis_2015.src.gaussian_embedding import DiagonalGaussianEmbedding


def test_diagonal_gaussian_embedding_shapes():
    model = DiagonalGaussianEmbedding(
        num_embeddings=10,
        embedding_dim=5
    )

    indices = torch.tensor([0, 1, 2])

    mu, logvar = model(indices)

    assert mu.shape == (3, 5)
    assert logvar.shape == (3, 5)


def test_diagonal_gaussian_embedding_parameters_require_grad():
    model = DiagonalGaussianEmbedding(
        num_embeddings=10,
        embedding_dim=5
    )

    assert model.mu.weight.requires_grad
    assert model.logvar.weight.requires_grad