import torch

from papers.p01_vilnis_2015.src.losses import margin_ranking_loss

def test_margin_ranking_loss_is_zero_when_margin_satisfied():
    positive_energy = torch.tensor([5.0])
    negative_energy = torch.tensor([1.0])

    loss = margin_ranking_loss(
        positive_energy,
        negative_energy,
        margin=1.0
    )

    assert torch.allclose(loss, torch.tensor(0.0))


def test_margin_ranking_loss_positive_when_margin_violated():
    positive_energy = torch.tensor([1.0])
    negative_energy = torch.tensor([1.5])

    loss = margin_ranking_loss(
        positive_energy,
        negative_energy,
        margin=1.0
    )

    assert loss > 0


def test_margin_ranking_loss_averages_batch():
    positive_energy = torch.tensor([5.0, 1.0])
    negative_energy = torch.tensor([1.0, 1.5])

    loss = margin_ranking_loss(
        positive_energy,
        negative_energy,
        margin=1.0
    )

    expected = torch.tensor((0.0 + 1.5) / 2.0)

    assert torch.allclose(loss, expected)