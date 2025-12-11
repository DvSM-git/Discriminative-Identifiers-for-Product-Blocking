import Levenshtein
import math

def q_gram_similarity(string_1, string_2, q):
    """
    Similarity of two q-grams (shingles)
    """
    q_grams_1 = {string_1[i:i+q] for i in range(len(string_1) - q + 1)}
    q_grams_2 = {string_2[i:i+q] for i in range(len(string_2) - q + 1)}

    intersect = q_grams_1.intersection(q_grams_2)
    union = q_grams_1.union(q_grams_2)

    return len(intersect) / len(union) if len(union) != 0 else 0

def cosineSim(string_1, string_2):
    """
    Cosine similarity between two strings
    """
    string_1_set = set(string_1.split())
    string_2_set = set(string_2.split())

    numerator = len(string_1_set.intersection(string_2_set))

    size_1 = len(string_1_set)
    size_2 = len(string_2_set)

    return numerator/(math.sqrt(size_1) * math.sqrt(size_2)) if size_1 != 0 and size_2 != 0 else 0


def norm_lv(string_1, string_2):
    """
    Levenshtein distance between two strings
    """
    lv_dist = Levenshtein.distance(string_1, string_2)
    max_len = max(len(string_1), len(string_2))

    return 0 if max_len == 0 else lv_dist/max_len


def avg_lv_sim(model_words_1, model_words_2, mw: bool):
    """
    Average Leneshtein distance between model word sets
    """
    numerator = 0
    denominator = 0

    for word_1 in model_words_1:
        non_numeric_1, numeric_1 = split_numeric(word_1)

        for word_2 in model_words_2:
            non_numeric_2, numeric_2 = split_numeric(word_2)

            if not mw or (mw and norm_lv(non_numeric_1, non_numeric_2) > 0.5 and numeric_1 == numeric_2):
                numerator += (1 - norm_lv(word_1,word_2)) * (len(word_1) + len(word_2))
                denominator += (len(word_1) + len(word_2))
    
    return numerator / denominator if denominator != 0 else 0

def split_numeric(string):
    """
    Splits strings into substrings
    """
    non_numeric = ''.join(filter(lambda char: not char.isdigit(), string))
    numeric = ''.join(filter(lambda char: char.isdigit(), string))
    
    return non_numeric, numeric