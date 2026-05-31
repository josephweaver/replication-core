import torch


def margin_ranking_loss(positive_energy, negative_energy, margin=1.0):
    """
    Vilnis-style max-margin ranking loss.

    We want positive pairs to have higher energy than negative pairs by at least margin.

    loss = max(0, margin - positive_energy + negative_energy)
    """

    loss_per_example = torch.clamp(
        margin - positive_energy + negative_energy,
        min=0.0
    )

    return torch.mean(loss_per_example)