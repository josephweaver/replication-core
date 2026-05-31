import torch

from papers.p01_vilnis_2015.src.energies import expected_likelihood_energy


def test_expected_likelihood_energy_shape():
    batch_size = 4
    embedding_dim = 8

    mu_i = torch.zeros(batch_size, embedding_dim)
    logvar_i = torch.zeros(batch_size, embedding_dim)

    mu_j = torch.ones(batch_size, embedding_dim)
    logvar_j = torch.zeros(batch_size, embedding_dim)

    energy = expected_likelihood_energy(
        mu_i,
        logvar_i,
        mu_j,
        logvar_j
    )

    assert energy.shape == (batch_size,)


def test_expected_likelihood_energy_is_symmetric():
    batch_size = 3
    embedding_dim = 5

    mu_i = torch.randn(batch_size, embedding_dim)
    logvar_i = torch.randn(batch_size, embedding_dim)

    mu_j = torch.randn(batch_size, embedding_dim)
    logvar_j = torch.randn(batch_size, embedding_dim)

    energy_ij = expected_likelihood_energy(
        mu_i,
        logvar_i,
        mu_j,
        logvar_j
    )

    energy_ji = expected_likelihood_energy(
        mu_j,
        logvar_j,
        mu_i,
        logvar_i
    )

    assert torch.allclose(energy_ij, energy_ji)


def test_expected_likelihood_energy_decreases_with_distance():
    embedding_dim = 4

    mu_origin = torch.zeros(1, embedding_dim)
    logvar_origin = torch.zeros(1, embedding_dim)

    mu_near = torch.ones(1, embedding_dim)
    mu_far = 10 * torch.ones(1, embedding_dim)

    logvar = torch.zeros(1, embedding_dim)

    energy_near = expected_likelihood_energy(
        mu_origin,
        logvar_origin,
        mu_near,
        logvar
    )

    energy_far = expected_likelihood_energy(
        mu_origin,
        logvar_origin,
        mu_far,
        logvar
    )

    assert energy_near > energy_far

