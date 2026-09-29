# From Noise to Signal: Discriminative Model Identifiers for Efficient Product Blocking

Finding the same product listed under different names in different web shops (entity resolution), with far fewer comparisons than checking every pair.

Solo project for the course *Computer Science for Business Analytics*, MSc Econometrics and Management Science, Erasmus School of Economics (December 2025). Python.

## The problem

The same television appears in several web shops with different titles, attribute names and formats. Comparing every pair of listings grows quadratically with the number of products. Locality-sensitive hashing (LSH) cuts the number of comparisons, but on noisy product titles it lets through many false candidates.

## Data

`TVs-all-merged.json`: 1,624 television listings from four web shops (Best Buy 773, Newegg 668, Amazon 163, TheNerds 20), describing 1,262 distinct products. Each listing has a title, a shop, a model identifier (used only as ground truth) and a dictionary of specifications whose keys and formats differ from shop to shop.

## Data preparation

- **Text cleaning.** Titles are lower-cased and units are standardised, so that `55"`, `55-inch` and `55 inches` become the same token (likewise for Hz and pounds).
- **Model words.** Alphanumeric identifiers such as `un40h6350` are extracted from titles and from the specification values, together with the brand. These identify a product far better than generic words.
- **Stop shingles.** Character substrings that appear in more than a set share of all titles (such as parts of "inch" or "led") carry no information and are removed.

## Method

1. **Features:** character shingles of each title, plus model words. Model words are repeated so they weigh more (**model word weighting**), and frequent shingles are pruned (**stop shingle pruning**).
2. **MinHash:** each listing is compressed into a short signature that preserves Jaccard similarity.
3. **LSH banding:** only listings that share a band of their signature become candidate pairs.
4. **Classification:** candidate pairs are scored with the Multi-Component Similarity Method (MSM), which combines the similarity of matching specifications, model words and titles, and then clustered.
5. **Tuning:** hyperparameters (shingle size, weights, thresholds, bands and rows) are tuned with Bayesian optimisation (Optuna), for the full model and for a restricted baseline without the two additions.

## Evaluation

Five bootstrap replications. In each one, the model is evaluated on the out-of-bag listings, and results are averaged across replications. Measures:

- **Pair quality and pair completeness:** precision and recall of the LSH candidate pairs; F1* combines them.
- **F1:** precision and recall of the final duplicate pairs after MSM.
- **Fraction of comparisons:** candidate pairs as a share of all possible pairs, i.e. the computational cost.

## Results

- **Accuracy:** the full model raises F1 by about 15% compared with the restricted baseline.
- **Efficiency:** it reaches its best performance while comparing only **1.2%** of all possible pairs, against 58.1% for the baseline.

Running `main.py` produces the performance-versus-cost plots for both models.

## Limitations

- Hyperparameters were tuned on the same bootstrap samples that are used to report the results, so the reported scores are somewhat optimistic. Tuning on the in-bag listings and evaluating on the out-of-bag ones would give a cleaner estimate.
- The stop-shingle list is computed from all listings rather than only the training listings.

## How to run

```
pip install -r requirements.txt
python main.py        # benchmark: full model vs restricted baseline, with plots
python Optimise.py    # optional: hyperparameter search (runs until stopped; saves progress to optuna_tv_study.db)
```

The tuned hyperparameters are already set in `main.py`, so `Optimise.py` is only needed to repeat the search.

## Repository structure

| File | Purpose |
|---|---|
| `preprocessing.py` | Text cleaning, unit standardisation, stop-shingle pruning, model-word extraction |
| `minhash.py` | MinHash signatures |
| `lsh.py` | LSH banding and candidate pairs |
| `msm.py` | Multi-Component Similarity Method and clustering |
| `main.py` | Bootstrap evaluation of the full and restricted models, plots |
| `Optimise.py` | Hyperparameter tuning with Optuna |
| `TVs-all-merged.json` | Data |
