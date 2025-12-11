# Discriminative-Identifiers-for-Product-Blocking

# From Noise to Signal: Leveraging Discriminative Model Identifiers for Efficient Product Blocking

This repository contains the implementation for the paper *From Noise to Signal: Leveraging Discriminative Model Identifiers for Efficient Product Blocking*. It provides an efficient Entity Resolution pipeline tailored for structured e-commerce data (e.g., televisions), addressing the low precision of standard Locality-Sensitive Hashing (LSH) on noisy datasets.

## Core Methodology
The pipeline enhances the Multi-Component Similarity Method (MSM) framework with two specific feature engineering optimisations:
1.  **Model Word Weighting:** Up-weights alphanumeric tokens (e.g., "UN40H6350") using token replication to prioritise discriminative identifiers.
2.  **Stop Shingle Pruning:** Filters high-frequency substrings (shingles) appearing in $>t_{freq}$ of descriptions to remove noise.

## Repository Structure
* **`preprocessing.py`**: Handles text normalisation, stop shingle pruning, and hybrid feature extraction.
* **`minhash.py`**: Implements MinHash signature generation using random linear permutations.
* **`lsh.py`**: Implements the LSH Banding technique for candidate generation.
* **`msm.py`**: Performs clustering using weighted similarity of attributes, model words, and titles.
* **`main.py`**: Executes the benchmark comparison between the Full Model and Restricted Baseline.
* **`Optimise.py`**: Performs Bayesian Optimisation (via Optuna) to tune hyperparameters (shingle size, bands, rows, weights.

## Key Results
The proposed Full Model significantly outperforms the standard baseline:
* **Efficiency:** Requires only **1.2%** of theoretical pairwise comparisons (vs 58.1% for baseline) to reach peak performance.
* **Effectiveness:** Increases the $F_1$ measure by approximately 15% compared to standard LSH implementations.

## Usage
1.  **Dependencies:** Requires `numpy`, `pandas`, `scikit-learn`, `networkx`, `joblib`, `optuna`, `ordered-set`, and `python-Levenshtein`.
2.  **Optimisation:** Run `python Optimise.py` to tune hyperparameters using the `TVs-all-merged.json` dataset.
3.  **Execution:** Run `python main.py` to generate performance metrics and comparison plots.
