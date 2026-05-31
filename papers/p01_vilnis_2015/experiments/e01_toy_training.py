import torch
from torch.optim import Adam

from papers.p01_vilnis_2015.src.gaussian_embedding import DiagonalGaussianEmbedding
from papers.p01_vilnis_2015.src.energies import expected_likelihood_energy
from papers.p01_vilnis_2015.src.losses import margin_ranking_loss


torch.manual_seed(1)

model = DiagonalGaussianEmbedding(num_embeddings=3, embedding_dim=2)
optimizer = Adam(model.parameters(), lr=0.05)

center = torch.tensor([0])
positive = torch.tensor([1])
negative = torch.tensor([2])

for step in range(200):
    optimizer.zero_grad()

    mu_c, logvar_c = model(center)
    mu_p, logvar_p = model(positive)
    mu_n, logvar_n = model(negative)

    positive_energy = expected_likelihood_energy(mu_c, logvar_c, mu_p, logvar_p)
    negative_energy = expected_likelihood_energy(mu_c, logvar_c, mu_n, logvar_n)

    loss = margin_ranking_loss(positive_energy, negative_energy, margin=1.0)

    loss.backward()
    optimizer.step()

    if step % 25 == 0:
        print(
            step,
            "loss=", loss.item(),
            "E_pos=", positive_energy.item(),
            "E_neg=", negative_energy.item()
        )