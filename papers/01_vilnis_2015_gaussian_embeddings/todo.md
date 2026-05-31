# TODO

## Phase 0: Read and extract math
- [ ] Write down expected likelihood formula
- [ ] Write down diagonal Gaussian KL formula
- [ ] Identify the training loss
- [ ] Identify which results require external datasets

## Phase 1: Minimal implementation
- [ ] Create `src/replication_core/`
- [ ] Create Gaussian embedding module
- [ ] Implement expected likelihood energy
- [ ] Implement KL divergence
- [ ] Write tests for shapes and finite values

## Phase 2: Toy training
- [ ] Create tiny corpus
- [ ] Generate positive center-context pairs
- [ ] Generate negative samples
- [ ] Train one tiny model
- [ ] Print nearest neighbors

## Phase 3: Paper-facing replication
- [ ] Compare variance behavior to paper claims
- [ ] Try KL scoring for entailment-like examples