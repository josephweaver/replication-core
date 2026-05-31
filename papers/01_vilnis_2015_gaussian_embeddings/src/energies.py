"""
Energy functions from Vilnis & McCallum (2015).

This file implements the mathematical similarity measures
used between Gaussian word representations.
"""
import torch


def expected_likelihood_energy(
    mu_i,
    logvar_i,
    mu_j,
    logvar_j
):
    """
    Compute the expected likelihood kernel energy between
    two diagonal Gaussians.

    Parameters
    ----------
    mu_i : tensor
    logvar_i : tensor
    mu_j : tensor
    logvar_j : tensor

    Returns
    -------
    energy : tensor
    """
