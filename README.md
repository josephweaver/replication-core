# replication-core

Goal: reproduce the core mathematical mechanisms of selected papers, not their full benchmark suites.

Core target:
- Identify primary claims.
- Implement an MVP to verify the identified claims.
- Attempt to use math, method, algorithms of the paper, or close proximities.
- Use subsets of large datasets when full datasets are impractical.

## Papers

1. (Vilnis 2015) Word Representations via Gaussian Embeddings

## Directory Structure

```text
replication-core/
├── README.md
├── .gitignore
├── common/
│   ├── README.md
│   ├── math/
│   ├── plotting/
│   └── utils/
├── papers/
│   ├── 01_<paper_name>/
│   │   ├── README.md
│   │   ├── <paper_name>.pdf
│   │   ├── paper_metadata.md
│   │   ├── notes.md
│   │   ├── requirements.txt
│   │   ├── requirements.lock.txt
│   │   ├── .venv/
│   │   ├── data/
│   │   │   ├── raw/
│   │   │   ├── interim/
│   │   │   └── processed/
│   │   ├── env/
│   │   |   └── create_venv.sh
│   │   ├── src/
│   │   │   └── *.py
│   │   ├── experiments/
│   │   ├── tests/
│   │   └── results/
│   ├── 02_next_paper/
│   └── ...
└── docs/
    ├── paper_index.md
    └── replication_protocol.md
```


## Project Conventions

- All scripts assume execution from the repository root.
- All paths should therefore be relative to the repository root.
- Python virtual environments are paper-local and stored in:
  `papers/<paper_name>/.venv`