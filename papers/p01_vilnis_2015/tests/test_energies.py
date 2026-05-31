import torch

from papers.p01_vilnis_2015.src.energies import expected_likelihood_energy, kl_divergence


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

def test_kl_divergence_shape():
    batch_size = 4
    embedding_dim = 8

    mu_p = torch.zeros(batch_size, embedding_dim)
    logvar_p = torch.zeros(batch_size, embedding_dim)

    mu_q = torch.ones(batch_size, embedding_dim)
    logvar_q = torch.zeros(batch_size, embedding_dim)

    kl = kl_divergence(mu_p, logvar_p, mu_q, logvar_q)

    assert kl.shape == (batch_size,)


def test_kl_divergence_is_zero_for_identical_gaussians():
    batch_size = 3
    embedding_dim = 5

    mu = torch.randn(batch_size, embedding_dim)
    logvar = torch.randn(batch_size, embedding_dim)

    kl = kl_divergence(mu, logvar, mu, logvar)

    assert torch.allclose(kl, torch.zeros(batch_size), atol=1e-6)


def test_kl_divergence_is_nonnegative():
    batch_size = 5
    embedding_dim = 7

    mu_p = torch.randn(batch_size, embedding_dim)
    logvar_p = torch.randn(batch_size, embedding_dim)

    mu_q = torch.randn(batch_size, embedding_dim)
    logvar_q = torch.randn(batch_size, embedding_dim)

    kl = kl_divergence(mu_p, logvar_p, mu_q, logvar_q)

    assert torch.all(kl >= -1e-6)


def test_kl_divergence_is_not_symmetric():
    batch_size = 4
    embedding_dim = 6

    mu_p = torch.randn(batch_size, embedding_dim)
    logvar_p = torch.randn(batch_size, embedding_dim)

    mu_q = torch.randn(batch_size, embedding_dim)
    logvar_q = torch.randn(batch_size, embedding_dim)

    kl_pq = kl_divergence(mu_p, logvar_p, mu_q, logvar_q)
    kl_qp = kl_divergence(mu_q, logvar_q, mu_p, logvar_p)

    assert not torch.allclose(kl_pq, kl_qp)