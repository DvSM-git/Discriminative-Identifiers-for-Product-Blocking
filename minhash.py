import numpy as np
import random
import zlib

class MinHasher:
    """
    Implements MinHash signature generation using random linear permutations
    """
    def __init__(self, num_perm=100, seed=0):

        self.num_perm = num_perm
        self.seed = seed
        self.prime = 2**32 - 1 
        self.permutations = self._generate_permutations()
        
    def _generate_permutations(self):
        """
        Generates K pairs of random coefficients (a, b) for the hash functions:
        h(x) = (a*x + b) % prime
        """
        random.seed(self.seed)
        max_val = 2**32 - 1
        
        perms = []
        for _ in range(self.num_perm):
            a = random.randint(1, max_val)
            b = random.randint(0, max_val)
            perms.append((a, b))
        return np.array(perms, dtype=np.uint64)

    def compute_signature(self, shingles): 
        """
        Computes the MinHash signature for a set of shingles
        """
        if not shingles:
            # Return a signature of max values if no shingles exist
            return np.full(self.num_perm, np.iinfo(np.uint64).max, dtype=np.uint64)

        # Hash shingles to 32-bit integers first
        shingle_hashes = np.array([zlib.crc32(s.encode('utf-8')) & 0xffffffff for s in shingles], dtype=np.uint64)

        
        # Reshape coefficients for broadcasting
        a = self.permutations[:, 0].reshape(-1, 1)
        b = self.permutations[:, 1].reshape(-1, 1)
        
        # Reshape inputs
        x = shingle_hashes.reshape(1, -1)
        
        # Compute h(x) for all combinations
        hashed_values = (np.matmul(a, x) + b) % self.prime
        
        # Find the minimum hash value for each permutation (row-wise min)
        signature = np.min(hashed_values, axis=1)
        
        return signature