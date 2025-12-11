from collections import defaultdict
import itertools

class LSHIndex:
    """
    Implements the Banding technique for LSH
    """
    def __init__(self, num_bands, rows_per_band, data_map=None):
        self.b = num_bands
        self.r = rows_per_band
        self.k = num_bands * rows_per_band
        self.buckets = [defaultdict(list) for _ in range(self.b)]
        self.data_map = data_map
        
    def insert(self, doc_id, signature):
        """
        Hashes the signature into buckets
        """
        if len(signature)!= self.k:
            raise ValueError(f"Signature length {len(signature)} does not match LSH params (b={self.b} * r={self.r} = {self.k})")
        
        for i in range(self.b):
            start = i * self.r
            end = start + self.r
            
            # Extract and hash the sub-vector
            band_vector = tuple(signature[start:end])
            bucket_id = hash(band_vector)
            self.buckets[i][bucket_id].append(doc_id)
            
    def get_candidates(self):
        """
        Retrieves all pairs of documents that share at least one bucket.
        """
        candidates = set()
        for band_idx in range(self.b):
            for bucket in self.buckets[band_idx].values():
                if len(bucket) > 1:
                    # If a bucket has >1 item, all combinations in it are candidates.
                    for pair in itertools.combinations(sorted(bucket), 2):
                        candidates.add(pair)
        return candidates   