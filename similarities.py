import math
import Levenshtein
from typing import List, Tuple, Set


def calculate_cosine_similarity(text_a: str, text_b: str) -> float:
    """
    Computes cosine similarity based on whitespace-separated tokens
    """
    tokens_a: Set[str] = set(text_a.split())
    tokens_b: Set[str] = set(text_b.split())

    mag_a = math.sqrt(len(tokens_a))
    mag_b = math.sqrt(len(tokens_b))
    
    if mag_a == 0 or mag_b == 0:
        return 0.0

    common_tokens = len(tokens_a & tokens_b)
    
    return common_tokens / (mag_a * mag_b)

def calculate_jaccard_kgram(text_a: str, text_b: str, k: int) -> float:
    """
    Calculates the Jaccard similarity index based on k-length shingles
    """
    shingles_a = {text_a[i : i + k] for i in range(len(text_a) - k + 1)}
    shingles_b = {text_b[i : i + k] for i in range(len(text_b) - k + 1)}

    intersection_count = len(shingles_a & shingles_b)
    union_count = len(shingles_a | shingles_b)

    return intersection_count / union_count if union_count > 0 else 0.0


def get_normalised_distance(s1: str, s2: str) -> float:
    """
    Returns the Levenshtein distance normalized by the maximum string length
    """
    max_char_len = max(len(s1), len(s2))
    
    if max_char_len == 0:
        return 0.0
        
    raw_distance = Levenshtein.distance(s1, s2)
    return raw_distance / max_char_len


def parse_alpha_numeric(text: str) -> Tuple[str, str]:
    """
    Separates a string into its non-numeric (alpha) and numeric components
    """
    alpha_part = "".join([char for char in text if not char.isdigit()])
    numeric_part = "".join([char for char in text if char.isdigit()])
    
    return alpha_part, numeric_part


def weighted_model_similarity(set_a: List[str], set_b: List[str], enforce_strict: bool) -> float:
    """
    Calculates a weighted average similarity between two sets of model words
    """
    accumulated_score = 0.0
    total_weight = 0.0

    for token_a in set_a:
        alpha_a, num_a = parse_alpha_numeric(token_a)

        for token_b in set_b:
            alpha_b, num_b = parse_alpha_numeric(token_b)

            process_pair = True
            
            if enforce_strict:
                dist_alpha = get_normalised_distance(alpha_a, alpha_b)
                if not (dist_alpha > 0.5 and num_a == num_b):
                    process_pair = False

            if process_pair:
                current_weight = len(token_a) + len(token_b)
                
                sim_score = 1.0 - get_normalised_distance(token_a, token_b)

                accumulated_score += sim_score * current_weight
                total_weight += current_weight

    return accumulated_score / total_weight if total_weight > 0 else 0.0