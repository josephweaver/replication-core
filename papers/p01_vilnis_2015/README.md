# Vilnis 2015 Replication Notes

This folder tracks what I understand, what I have implemented, and what evidence would count as replication.

Replication levels:

1. Mathematical replication:
   - derive expected likelihood energy
   - derive KL divergence between diagonal Gaussians

2. Software replication:
   - implement Gaussian embeddings
   - implement energy functions
   - train on toy corpus

3. Empirical replication:
   - check nearest neighbors
   - check variance behavior
   - later compare to entailment results


## Checkpoint 1: Toy Training

A three-item toy model was trained with one center item, one positive item, and one negative item.

Result after training:

- loss: 0.0
- positive energy: 0.2336
- negative energy: -2.3451

Interpretation:

The expected-likelihood energy and margin ranking loss are sufficient to train a diagonal Gaussian embedding model on a minimal toy example.