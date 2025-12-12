import os
import itertools
import matplotlib.pyplot as plt
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import itertools
from collections import defaultdict
import numpy as np
from sklearn.utils import resample
import pandas as pd
import time
from joblib import Parallel, delayed
from preprocessing import HybridPreprocessor
from minhash import MinHasher
from lsh import LSHIndex
from msm import MSMClassifier

def calculate_comprehensive_metrics(lsh_candidates, msm_clusters, true_pairs, n_total_docs):
    """
    Computes metrics for both LSH (candidates) and MSM (final predictions)
    """
    n_duplicates = len(true_pairs)
    
    df_set = lsh_candidates.intersection(true_pairs)
    df = len(df_set)
    n_comp = len(lsh_candidates)
    
    # Pair Quality (LSH Precision)
    pq = df / n_comp if n_comp > 0 else 0.0
    
    # Pair Completeness (LSH Recall)
    pc = df / n_duplicates if n_duplicates > 0 else 0.0
    
    # F1* (LSH F1)
    if (pq + pc) == 0:
        f1_star = 0.0
    else:
        f1_star = 2 * pq * pc / (pq + pc)
        
    # Fraction of Comparisons
    n_possible = n_total_docs * (n_total_docs - 1) / 2
    frac_comp = n_comp / n_possible if n_possible > 0 else 0.0

    # F1
    msm_predictions = set()
    for cluster in msm_clusters:
        if len(cluster) > 1:
            for pair in itertools.combinations(sorted(cluster), 2):
                msm_predictions.add(pair)
                
    tp_set = msm_predictions.intersection(true_pairs)
    tp = len(tp_set)
    
    fp = len(msm_predictions) - tp
    fn = n_duplicates - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if (precision + recall) == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
        
    return f1, f1_star, pc, pq, n_comp, frac_comp

def get_true_pairs_for_subset(products, indices):
    """
    Gets the true duplicates for a given subset of products
    """
    subset_products = [products[i] for i in indices]
    model_groups = defaultdict(list)
    for local_idx, p in enumerate(subset_products):
        model_groups[p['true_model_id']].append(local_idx)
        
    true_pairs = set()
    for ids in model_groups.values():
        if len(ids) > 1:
            for pair in itertools.combinations(sorted(ids), 2):
                true_pairs.add(pair)
    return true_pairs

def run_single_bootstrap(b, all_products, precomputed_features, all_indices, n_total, config):
    """
    Executes a single bootstrap iteration.
    """
    # LSH Config
    NUM_PERM = config['NUM_PERM']
    BANDS = config['BANDS']
    ROWS = config['ROWS']
    
    # MSM Config
    THRESHOLD = config['THRESHOLD']
    GAMMA = config['GAMMA']
    EPSILON = config['EPSILON']
    MU = config['MU']
    ALPHA = config['ALPHA']
    BETA = config['BETA']
    DELTA = config['DELTA']
    
    # Resampling
    boot_indices = resample(all_indices, replace=True, n_samples=n_total, random_state=b)
    
    # OOB Calculation
    boot_set = set(boot_indices)
    oob_indices = sorted(list(set(all_indices) - boot_set))
    
    if len(oob_indices) < 2:
        return (0.0, 0.0, 0.0, 0.0, b)

    current_n = len(oob_indices)
    
    # Mappings
    id_to_product_map = {
        local_idx: all_products[global_idx] 
        for local_idx, global_idx in enumerate(oob_indices)
    }

    # Ground Truth
    true_pairs = get_true_pairs_for_subset(all_products, oob_indices)
    
    # Pipeline
    minhasher = MinHasher(num_perm=NUM_PERM, seed=b)
    lsh = LSHIndex(num_bands=BANDS, rows_per_band=ROWS, data_map=id_to_product_map)
    
    for local_idx, global_idx in enumerate(oob_indices):
        features = precomputed_features[global_idx]
        sig = minhasher.compute_signature(features)
        lsh.insert(local_idx, sig)
        
    candidates = lsh.get_candidates()
    
    # Clustering
    msm = MSMClassifier(
        id_to_product_map, 
        gamma=GAMMA, 
        epsilon=EPSILON, 
        mu=MU, 
        alpha=ALPHA, 
        beta=BETA, 
        delta=DELTA, 
        threshold=THRESHOLD
    )
    clusters = msm.cluster(candidates, id_to_product_map)
    
    # Calculate Comprehensive Metrics
    f1, f1_star, pc, pq, n_comp, frac_comp = calculate_comprehensive_metrics(
        candidates, 
        clusters, 
        true_pairs, 
        current_n
    )
    
    return (f1, f1_star, pc, pq, n_comp, frac_comp, b)


def run_grid_search(json_file, param_grid):
    """
    Runs a grid search schema if Bayesian search is not wanted
    """
    # Load data
    print("Loading data...")
    loader = HybridPreprocessor(json_file)
    all_products = loader.load_data()
    n_total = len(all_products)
    all_indices = np.arange(n_total)
    print(f"Loaded {len(all_products)} products.")

    # Generate all parameter combinations
    combinations = list(itertools.product(
        param_grid['LSH_PARAMS'],
        param_grid['THRESHOLD'],
        param_grid['SHINGLE_SIZE'],
        param_grid['MW_WEIGHT'],
        param_grid['STOP_FREQ'],
        param_grid['GAMMA'],
        param_grid['EPSILON'],
        param_grid['MU'],
        param_grid['ALPHA'],
        param_grid['BETA'],
        param_grid['DELTA']
    ))
    
    # Sort by Preprocessing Params to Group Feature Computation
    combinations.sort(key=lambda x: (x[2], x[3], x[4])) 
    
    print(f"--- Strategy: Parallel Grid Search (Optimized Grouping) ---")
    print(f"Testing {len(combinations)} configurations across {param_grid['NUM_BOOTSTRAPS']} bootstraps.")
    
    start_time = time.time()
    final_results = []

    # Iterate through Preprocessing Groups
    for (s_size, mw_w, stop_f), group in itertools.groupby(combinations, key=lambda x: (x[2], x[3], x[4])):

        # Update global Preprocessor state
        loader.compute_shingle_stats(all_products, k=s_size, threshold_freq=stop_f)

        print(f"\nPre-computing features for Shingle={s_size}, MW_Weight={mw_w}, StopFreq={stop_f}...")
        
        # Pre-compute features once per group
        precomputed_features = [
            loader.get_hybrid_features(product, k=s_size, mw_weight=mw_w) 
            for product in all_products
        ]
        
        # Build Task List for this Group
        tasks = []
        group_list = list(group)
        
        for (bands, rows), thr, _, _, _, gam, eps, mu, alp, bet, delt in group_list:
            config = {
                'NUM_PERM': bands * rows,
                'BANDS': bands,
                'ROWS': rows,
                'THRESHOLD': thr,
                'SHINGLE_SIZE': s_size,
                'MW_WEIGHT': mw_w,
                'STOP_FREQ': stop_f,
                'GAMMA': gam,
                'EPSILON': eps,
                'MU': mu,
                'ALPHA': alp,
                'BETA': bet,
                'DELTA': delt
            }
            # Create a task for every bootstrap seed
            for b in range(param_grid['NUM_BOOTSTRAPS']):
                tasks.append((config, b))
        
        print(f"  > Queuing {len(tasks)} tasks on all available cores...")

        # Execute all tasks in parallel
        raw_results = Parallel(n_jobs=-1, verbose=5)(
            delayed(run_single_bootstrap)(
                b, all_products, precomputed_features, all_indices, n_total, config
            )
            for config, b in tasks
        )
        
        # Aggregate Results
        agg_map = defaultdict(list)
        
        for idx, (f1, f1_star, pc, pq, n_comp, frac, b) in enumerate(raw_results):
            config, _ = tasks[idx]
            
            # Key now includes ALL variable params
            cfg_key = (
                config['BANDS'], config['ROWS'], config['THRESHOLD'], config['SHINGLE_SIZE'],
                config['MW_WEIGHT'], config['STOP_FREQ'], config['GAMMA'], config['EPSILON'],
                config['MU'], config['ALPHA'], config['BETA'], config['DELTA']
            )
            
            agg_map[cfg_key].append({
                'f1': f1, 'f1_star': f1_star, 'pq': pq, 'pc': pc, 'frac': frac
            })
            
        # Calculate Averages
        for params, metrics in agg_map.items():
            (bnd, row, thr, s_size, mw, stop, gam, eps, mu, alp, bet, delt) = params
            
            mean_f1 = np.mean([m['f1'] for m in metrics])
            mean_f1_star = np.mean([m['f1_star'] for m in metrics])
            mean_pq = np.mean([m['pq'] for m in metrics])
            mean_pc = np.mean([m['pc'] for m in metrics])
            mean_frac = np.mean([m['frac'] for m in metrics])
            
            final_results.append([
                bnd, row, bnd*row, thr, s_size, mw, stop, gam, eps, mu, alp, bet, delt,
                mean_f1, mean_f1_star, mean_pq, mean_pc, mean_frac
            ])

    columns = [
        "Bands", "Rows", "Signatures", "Threshold", 
        "Shingle_Size", "MW_Weight", "Stop_Freq",
        "Gamma", "Epsilon", "Mu", "Alpha", "Beta", "Delta",
        "Mean_F1","Mean_F1_star", "Mean_PQ", "Mean_PC", "Frac_Comps"
    ]
    
    output = pd.DataFrame(final_results, columns=columns)
    output = output.sort_values(by="Mean_F1_star", ascending=False)
    
    print(f"Grid Search Completed in {time.time() - start_time:.2f} seconds.")
    return output


def plot_performance_vs_cost(df, output_filename="f1_vs_comparisons.png"):
    """
    Saves a plot of performance vs fraction of comparisons
    """
    plt.figure(figsize=(10, 6))
    df_sorted = df.sort_values(by="Frac_Comps")
    plt.plot(df_sorted["Frac_Comps"], df_sorted["Mean_F1"], linestyle='-', linewidth=2, label='MSM F1 (Final)')
    plt.plot(df_sorted["Frac_Comps"], df_sorted["Mean_F1_star"], linestyle='--', linewidth=2, label='LSH F1* (Candidates)')
    plt.xlabel('Fraction of Comparisons (Computational Cost)', fontsize=12)
    plt.ylabel('Performance Score (0-1)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to {output_filename}")
    plt.close()

def plot_combined_performance(df_full, df_restricted, output_filename="f1_comparison_combined.png"):
    """
    Plots the performance of both the Full and Restricted models on the same graph
    """
    plt.figure(figsize=(10, 6))
    
    # Sort dataframes
    df_full_srtd = df_full.sort_values(by="Frac_Comps")
    df_restr_srtd = df_restricted.sort_values(by="Frac_Comps")
    
    # Plot Full Model
    plt.plot(df_full_srtd["Frac_Comps"], df_full_srtd["Mean_F1"], 
             linestyle='-', linewidth=2, color='#1f77b4', label='Full Model (MSM F1)')
             
    # Plot Restricted Model
    plt.plot(df_restr_srtd["Frac_Comps"], df_restr_srtd["Mean_F1"], 
             linestyle='-', linewidth=2, color='#d62728', label='Restricted Model (MSM F1)')

    # Optional: Plot F1* (Candidates) for context
    plt.plot(df_full_srtd["Frac_Comps"], df_full_srtd["Mean_F1_star"], 
             linestyle=':', linewidth=1, color='#1f77b4', alpha=0.5, label='Full (LSH F1*)')
    plt.plot(df_restr_srtd["Frac_Comps"], df_restr_srtd["Mean_F1_star"], 
             linestyle=':', linewidth=1, color='#d62728', alpha=0.5, label='Restricted (LSH F1*)')

    plt.xlabel('Fraction of Comparisons (Computational Cost)', fontsize=12)
    plt.ylabel('Performance Score (0-1)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.savefig(output_filename, dpi=300)
    print(f"Combined plot saved to {output_filename}")
    plt.close()

if __name__ == "__main__":
    file_path = "TVs-all-merged.json" 

    TARGET_SIGS = 200 
    
    valid_lsh_pairs_smooth = [
        (b, int(TARGET_SIGS / b)) 
        for b in range(1, TARGET_SIGS + 1) 
        if b != 1 and int(TARGET_SIGS / b) != 1
    ]

    valid_lsh_pairs_smooth = sorted(list(set(valid_lsh_pairs_smooth)))

    print(f"Testing these (Band, Row) pairs: {valid_lsh_pairs_smooth}")



    print(f"Testing these (Band, Row) pairs: {valid_lsh_pairs_smooth}")


    # Optimal hyperparameters for the full model
    settings_grid_full = {
        'NUM_BOOTSTRAPS': 5,
        'LSH_PARAMS': valid_lsh_pairs_smooth,
        'THRESHOLD': [0.528],
        'SHINGLE_SIZE': [4],
        'MW_WEIGHT': [2],
        'STOP_FREQ': [0.0138],
        'GAMMA': [0.718],
        'EPSILON': [0.454],
        'MU': [0.819],
        'ALPHA': [0.558],
        'BETA': [0.859],
        'DELTA': [0.190]
    }

    # Optimal hyperparameters for the restricted model
    settings_grid_restricted = {
        'NUM_BOOTSTRAPS': 5,
        'LSH_PARAMS': valid_lsh_pairs_smooth,
        'THRESHOLD': [0.618],
        'SHINGLE_SIZE': [5],
        'MW_WEIGHT': [0],
        'STOP_FREQ': [1],
        'GAMMA': [0.744],
        'EPSILON': [0.381],
        'MU': [0.559],
        'ALPHA': [0.567],
        'BETA': [0.611],
        'DELTA': [0.215]
    }

    print("\n=== RUNNING FULL MODEL ===")
    df_full = run_grid_search(file_path, settings_grid_full)
    print("\nTop 10 Configurations for full model:")
    print(df_full)
    plot_performance_vs_cost(df_full, "f1_vs_comparisons_full.png")

    print("\n=== RUNNING RESTRICTED MODEL ===")
    df_restricted = run_grid_search(file_path, settings_grid_restricted)
    print("\nTop 10 Configurations for restricted model:")
    print(df_restricted)
    plot_performance_vs_cost(df_restricted, "f1_vs_comparisons_restricted.png")
    
    print("\n=== GENERATING COMBINED PLOT ===")
    plot_combined_performance(df_full, df_restricted, "f1_comparison_combined.png")
