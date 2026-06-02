"""
Energy functions for Vilnis & McCallum (2015).

The goal of this file is to implement the mathematical scoring
functions between diagonal Gaussian embeddings.
"""

import torch


def expected_likelihood_energy(mu_i, logvar_i, mu_j, logvar_j):
    """
    Compute log N(0; mu_i - mu_j, Sigma_i + Sigma_j)
    for diagonal covariance Gaussians.

    All inputs should have shape:

        (batch_size, embedding_dim)

    Returns
    -------
    torch.Tensor
        Shape: (batch_size,)
    """

    var_i = torch.exp(logvar_i)
    var_j = torch.exp(logvar_j)

    var_sum = var_i + var_j
    diff = mu_i - mu_j

    log_det_term = torch.log(var_sum)
    quadratic_term = diff**2 / var_sum

    energy_per_dim = log_det_term + quadratic_term

    energy = -0.5 * torch.sum(energy_per_dim, dim=1)

    return energy

def kl_divergence(mu_p, logvar_p, mu_q, logvar_q):
    """
    Compute KL(P || Q) for diagonal Gaussians.

    P = N(mu_p, Sigma_p)
    Q = N(mu_q, Sigma_q)

    Inputs have shape:

        (batch_size, embedding_dim)

    Returns
    -------
    torch.Tensor
        Shape: (batch_size,)
    """

    var_p = torch.exp(logvar_p)
    var_q = torch.exp(logvar_q)

    diff = mu_q - mu_p

    log_det_ratio = logvar_q - logvar_p
    trace_term = var_p / var_q
    quadratic_term = diff**2 / var_q

    kl_per_dim = log_det_ratio + trace_term + quadratic_term - 1.0

    kl = 0.5 * torch.sum(kl_per_dim, dim=1)

    return kl