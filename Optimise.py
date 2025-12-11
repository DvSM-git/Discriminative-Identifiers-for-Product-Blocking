import os
import optuna
import numpy as np
import itertools
from joblib import Parallel, delayed
from sklearn.utils import resample
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from preprocessing import HybridPreprocessor
from minhash import MinHasher
from lsh import LSHIndex
from msm import MSMClassifier
from main import calculate_comprehensive_metrics, get_true_pairs_for_subset, resolve_transitivity


def run_trial_bootstrap(b, all_products, all_indices, n_total, target_sig_len, params, features):
    """
    Worker function for running a single bootstrap
    """
    boot_indices = resample(all_indices, replace=True, n_samples=n_total, random_state=b)
    
    # OOB Calculation
    boot_set = set(boot_indices)
    oob_indices = sorted(list(set(all_indices) - boot_set))
    
    if len(oob_indices) < 2: return 0.0

    # Mappings
    id_to_product_map = {
        local_idx: all_products[global_idx] 
        for local_idx, global_idx in enumerate(oob_indices)
    }

    # Ground Truth
    true_pairs = get_true_pairs_for_subset(all_products, oob_indices)
    
    # Pipeline Execution
    minhasher = MinHasher(num_perm=target_sig_len, seed=b)
    
    lsh = LSHIndex(
        num_bands=params['bands'], 
        rows_per_band=params['rows'], 
        data_map=id_to_product_map
    )
    
    for local_idx, global_idx in enumerate(oob_indices):
        f = features[global_idx]
        sig = minhasher.compute_signature(f)
        lsh.insert(local_idx, sig)
        
    candidates = lsh.get_candidates()
    
    # MSM Clustering
    msm = MSMClassifier(
        id_to_product_map,
        gamma=params['gamma'],
        epsilon=params['epsilon'],
        mu=params['mu'],
        alpha=params['alpha'],
        beta=params['beta'],
        delta=params['delta'],
        threshold=params['threshold']
    )
    
    clusters = msm.cluster(candidates, id_to_product_map)
    
    # Metrics
    raw_pred_pairs = set()
    for cluster in clusters:
        for pair in itertools.combinations(sorted(cluster), 2):
            raw_pred_pairs.add(pair)
            
    refined_pred_pairs = resolve_transitivity(raw_pred_pairs)

    f1, _, _, _, _, _ = calculate_comprehensive_metrics(candidates, clusters, true_pairs, len(oob_indices))    
    
    return f1


class ProductMatchingObjective:
    """
    Optuna objective function
    """
    def __init__(self, json_file, target_signature_len=200, n_bootstraps=5):
        self.json_file = json_file
        self.target_sig_len = target_signature_len
        self.n_bootstraps = n_bootstraps
        
        print("Loading data for Optuna...")
        self.preprocessor = HybridPreprocessor(json_file)
        self.all_products = self.preprocessor.load_data()
        self.n_total = len(self.all_products)
        self.all_indices = np.arange(self.n_total)
        
        self.feature_cache = {}
        
        self.valid_lsh_pairs = [
            (b, int(target_signature_len/b)) 
            for b in range(1, target_signature_len + 1) 
            if target_signature_len % b == 0
        ]

    def get_features(self, shingle_size, mw_weight, threshold_freq):
        """
        Retrieves features, re-computing only if preprocessing params change
        """
        cache_key = (shingle_size, mw_weight, threshold_freq)
        
        if cache_key in self.feature_cache:
            return self.feature_cache[cache_key]
        
        print(f"[Cache Miss] Computing: k={shingle_size}, mw={mw_weight}, stop_freq={threshold_freq:.3f}")
        
        # Update shingle stats (This determines which shingles are 'stop words')
        self.preprocessor.compute_shingle_stats(
            self.all_products, 
            k=shingle_size, 
            threshold_freq=threshold_freq
        )
        
        # Compute features using the updated stop list
        features = [
            self.preprocessor.get_hybrid_features(p, k=shingle_size, mw_weight=mw_weight)
            for p in self.all_products
        ]
        
        self.feature_cache[cache_key] = features
        return features

    def __call__(self, trial):
        shingle_size = trial.suggest_int('shingle_size', 2, 8)

        # For full model
        mw_weight = trial.suggest_int('mw_weight', 1, 16)

        # For restricted model
        # mw_weight = 0

        
        # For full model
        stop_shingle_freq = trial.suggest_float('stop_shingle_freq', 0, 0.1)

        # For restricted model
        # stop_shingle_freq = 1


        lsh_idx = trial.suggest_categorical('lsh_pair_index', list(range(len(self.valid_lsh_pairs))))
        bands, rows = self.valid_lsh_pairs[lsh_idx]
        
        params = {
            'bands': bands,
            'rows': rows,
            'gamma': trial.suggest_float('gamma', 0.5, 0.95),
            'epsilon': trial.suggest_float('epsilon', 0.1, 0.9),
            'mu': trial.suggest_float('mu', 0.1, 0.9),
            'alpha': trial.suggest_float('alpha', 0.1, 0.9),
            'beta': trial.suggest_float('beta', 0.1, 0.9),
            'delta': trial.suggest_float('delta', 0.1, 0.9),
            'threshold': trial.suggest_float('threshold', 0.1, 0.9)
        }

        # Get Features
        features = self.get_features(shingle_size, mw_weight, stop_shingle_freq)
        
        # Run Bootstraps
        scores = Parallel(n_jobs=-1)(
            delayed(run_trial_bootstrap)(
                b, 
                self.all_products, 
                self.all_indices, 
                self.n_total, 
                self.target_sig_len, 
                params, 
                features
            )
            for b in range(self.n_bootstraps)
        )

        return np.mean(scores)

if __name__ == "__main__":
    file_path = "TVs-all-merged.json" 
    
    objective = ProductMatchingObjective(file_path, n_bootstraps=5)
    

    # Name the database file
    db_file_name = "optuna_tv_study.db"
    storage_url = f"sqlite:///{db_file_name}"

    # Give the study a unique name (required for resuming)
    study_name = "tv_matching_experiment_V3"

    print("Starting Optimization with Persistent Storage...")

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="maximize"
    )
    
    # n_jobs=1 because we use Parallel inside the trial
    study.optimize(objective, n_trials=None, n_jobs=1)
    
    print("\n--- Best Trial ---")
    print(f"F1 Score: {study.best_value}")
    print(study.best_params)