import numpy as np
import re
from collections import defaultdict
from ordered_set import OrderedSet
from sklearn.cluster import AgglomerativeClustering
from functools import lru_cache
from similarities import *

# Wrapper for caching external function calls
@lru_cache(maxsize=10000)
def cached_q_gram(s1, s2, q=3):
    return q_gram_similarity(string_1=s1, string_2=s2, q=q)

class MSMClassifier:
    """
    Optimised Multi-Component Similarity Method (MSM) for product deduplication
    """
    def __init__(self, brands_dict, gamma=0.718, epsilon=0.454, mu=0.819, 
                 alpha=0.558, beta=0.859, delta=0.190, threshold=0.5):
        if isinstance(brands_dict, dict):
            self.brands = set(str(v).lower() for v in brands_dict.values())
        else:
            self.brands = set(str(v).lower() for v in brands_dict)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.threshold = threshold
        
        # Pre-compile Regex Patterns
        self.res_regex = re.compile(r'(?<!\S)\d{3,}\s*[x]\s*\d{3,}(?!\S)', re.IGNORECASE)
        self.mw_regex = re.compile(r'^\d+\.\d+|\b\d+:\d+\b|(?<!\S)\d{3,}\s*[x]\s*\d{3,}(?!\S)')
        self.title_mw_regex = re.compile(r'([a-zA-Z0-9]*((\d*\.)?\d+[^0-9, ]+)[a-zA-Z0-9]*)')

    def cluster(self, candidate_pairs, data_map):
        """
        Main method to compute distances for candidates and cluster them
        """
        # Identify unique items involved in candidates
        unique_ids = sorted(list(set(x for pair in candidate_pairs for x in pair)))
        n = len(unique_ids)
        
        if n == 0:
            return []

        # Map global ID -> Local Matrix ID
        id_map = {global_id: local_id for local_id, global_id in enumerate(unique_ids)}
        
        # Pre-process data
        processed_data = self._pre_process_data(unique_ids, data_map)

        results = []
        for p1, p2 in candidate_pairs:
            if p1 not in id_map or p2 not in id_map:
                continue
            
            # Retrieve pre-processed objects
            obj1 = processed_data[p1]
            obj2 = processed_data[p2]
            
            sim = self._compute_similarity(obj1, obj2)
            results.append((id_map[p1], id_map[p2], 1 - sim))

        # Initialise Distance Matrix
        dist_matrix = np.ones((n, n))
        np.fill_diagonal(dist_matrix, 0)

        # Fill Matrix
        for res in results:
            if res is None: continue
            r, c, dist = res
            dist_matrix[r, c] = dist
            dist_matrix[c, r] = dist

        # Perform Agglomerative Clustering
        clustering = AgglomerativeClustering(
            metric="precomputed", 
            linkage="average", 
            distance_threshold=self.epsilon, 
            n_clusters=None
        )
        clustering.fit(dist_matrix)
        
        # Extract Clusters
        labels = clustering.labels_
        clusters_dict = defaultdict(set)
        for local_idx, label in enumerate(labels):
            global_id = unique_ids[local_idx]
            clusters_dict[label].add(global_id)
            
        return [c for c in clusters_dict.values() if len(c) > 1]

    def _pre_process_data(self, unique_ids, data_map):
        """
        Extracts Brand, Resolution, and MW once per product
        """
        processed = {}
        for pid in unique_ids:
            item = data_map[pid]
            features = item.get("featuresMap", {})
            title = item.get("title", "")
            shop = item.get("shop", "")

            # Extract Brand
            brand = features.get("Brand", "na").lower()
            if brand == "na":
                for b_name in self.brands:
                    if b_name in title.lower(): 
                         if re.search(rf'\b{re.escape(b_name)}\b', title.lower()):
                            brand = b_name
                            break
            
            # Extract Resolution
            res = "NA"
            for k, v in features.items():
                if "resolution" in k.lower():
                    match = self.res_regex.search(str(v))
                    if match:
                        res = match.group(0)
                        break
            
            # Extract MWs for Feature Matching            
            processed[pid] = {
                "features": features,
                "title": title,
                "shop": shop,
                "brand": brand,
                "resolution": res,
                "keys_set": set(features.keys())
            }
        return processed

    def _compute_similarity(self, p1, p2):
        """
        Computes similarity using pre-processed data objects
        """
        # Binary Checks (Pre-filtering)
        if p1['shop'] == p2['shop'] and p1['shop'] != "":
            return 0.0
        if p1['brand'] != p2['brand'] and p1['brand'] != "na" and p2['brand'] != "na":
            return 0.0
        if p1['resolution'] != p2['resolution'] and p1['resolution'] != "NA" and p2['resolution'] != "NA":
            return 0.0

        # Key-Value Matching
        sim = 0
        w = 0
        m = 0
        
        features_1 = p1['features']
        features_2 = p2['features']
        
        unmatched_keys_2 = p2['keys_set'].copy()
        matched_keys_1 = set()

        for key_1 in features_1:
            best_match_key = None
            
            # Greedy matching against remaining keys in p2
            for key_2 in unmatched_keys_2:
                # Use cached similarity
                key_sim = cached_q_gram(key_1, key_2, q=3)
                
                if key_sim > self.gamma:
                    val_1 = str(features_1[key_1])
                    val_2 = str(features_2[key_2])
                    value_sim = cached_q_gram(val_1, val_2, q=3)
                    
                    sim += key_sim * value_sim
                    w += key_sim
                    m += 1
                    
                    best_match_key = key_2
                    matched_keys_1.add(key_1)
                    break # Stop looking for this key_1
            
            if best_match_key:
                unmatched_keys_2.remove(best_match_key)

        mean_sim = sim / w if w > 0 else 0

        # Model Words calculation
        unmatched_keys_1 = p1['keys_set'] - matched_keys_1
        
        mw_1 = self._extract_model_words(features_1, unmatched_keys_1)
        mw_2 = self._extract_model_words(features_2, unmatched_keys_2)
        
        # Use standard set intersection for speed
        mw_1_set = set(mw_1)
        mw_2_set = set(mw_2)
        
        union_len = len(mw_1_set | mw_2_set)
        mw_percentage = len(mw_1_set & mw_2_set) / union_len if union_len > 0 else 0

        # Title Comparison
        title_sim = self._title_comp(p1['title'], p2['title'])

        # Weighted aggregation
        min_features = min(len(features_1), len(features_2))
        min_features = 1 if min_features == 0 else min_features 
        
        if title_sim == -1:
            theta_1 = m / min_features
            theta_2 = 1 - theta_1
            h_sim = theta_1 * mean_sim + theta_2 * mw_percentage
        else:
            theta_1 = (1 - self.mu) * (m / min_features)
            theta_2 = 1 - self.mu - theta_1
            h_sim = theta_1 * mean_sim + theta_2 * mw_percentage + self.mu * title_sim
            
        return h_sim

    def _extract_model_words(self, features, keys):
        """
        Extracts model wordsn from a document
        """
        mw = OrderedSet()
        for key in keys:
            val = features.get(key)
            if val:
                matches = self.mw_regex.findall(str(val))
                mw.update(matches)
        return mw

    def _title_comp(self, t1, t2):
        """
        Computes similarity between titles
        """
        name_cosine_sim = cosineSim(t1, t2)
        if name_cosine_sim > self.alpha:
            return 1

        # Use pre-compiled regex
        mw1 = OrderedSet(x[0] for x in self.title_mw_regex.findall(t1))
        mw2 = OrderedSet(x[0] for x in self.title_mw_regex.findall(t2))

        similar_mw = False
        
        for w1 in mw1:
            nn1, num1 = split_numeric(w1)
            for w2 in mw2:
                nn2, num2 = split_numeric(w2)
                
                if num1 != num2:
                    continue 

                threshold_sim = norm_lv(nn1, nn2)
                if threshold_sim > self.threshold:
                    similar_mw = True
                    pass 

        for w1 in mw1:
            nn1, num1 = split_numeric(w1)
            for w2 in mw2:
                nn2, num2 = split_numeric(w2)
                threshold_sim = norm_lv(nn1, nn2)
                if threshold_sim > self.threshold:
                    if num1 != num2: 
                        return -1
                    else: 
                        similar_mw = True

        final_sim = self.beta * name_cosine_sim + (1 - self.beta) * avg_lv_sim(mw1, mw2, mw=False)
        
        if similar_mw:
            final_sim = self.delta * avg_lv_sim(mw1, mw2, mw=True) + (1 - self.delta) * final_sim
            
        return final_sim