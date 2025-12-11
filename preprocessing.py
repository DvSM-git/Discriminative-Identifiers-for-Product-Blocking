import json
import re
from collections import Counter

class HybridPreprocessor:
    """
    Implements the preprocessing of the JSON file to extract pruned shingles and model words
    """
    def __init__(self, json_path):
        self.json_path = json_path
        self.products = []
        self.title_regex = re.compile(r'([a-zA-Z0-9]*((\d*\.)?\d+[^0-9, ]+)[a-zA-Z0-9]*)')
        self.kvp_regex = re.compile(r'^\d+\.\d+|\b\d+:\d+\b|(?<!\S)\d{3,}\s*[x]\s*\d{3,}(?!\S)')
        
        self.stop_shingles = set()

    def load_data(self):
        """
        Loads and processes the JSON file
        """
        with open(self.json_path, 'r') as f:
            raw_data = json.load(f)
            
        processed_list = []
        for model_id, product_list in raw_data.items():
            for product in product_list:
                product['true_model_id'] = model_id 
                processed_list.append(product)
        
        self.products = processed_list
        return self.products

    def clean_text(self, text):
        """
        Normalises text and standardises units
        """

        if not text: return ""
        text = text.lower()
    
        
        # Standardising units
        text = re.sub(r'(\d+)\s*[- ]?\s*(?:inch(?:es)?|")', r'\1inch', text)
        text = re.sub(r'(\d+)\s*[- ]?\s*(?:hz|hertz)', r'\1hz', text)
        text = re.sub(r'(\d+)\s*[- ]?\s*(?:lbs|pounds)', r'\1lbs', text)
        text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
        
        tokens = text.split()

        return " ".join(tokens).strip()

    def compute_shingle_stats(self, products, k=3, threshold_freq=0.05):
        """
        Identifies shingles that appear in more than 'threshold_freq' fraction
        of the documents.
        """
        print(f"Computing shingle statistics (k={k})...")
        doc_counts = Counter()
        n_docs = len(products)
        
        for p in products:
            # Generate shingles temporarily just to count them
            text = self.clean_text(p.get("title", ""))
            if len(text) < k:
                continue
            
            # Use a set to count presence per document (binary freq), not raw term freq
            unique_shingles_in_doc = {text[i : i+k] for i in range(len(text) - k + 1)}
            doc_counts.update(unique_shingles_in_doc)
            
        cutoff = n_docs * threshold_freq
        self.stop_shingles = {s for s, count in doc_counts.items() if count > cutoff}
        
        print(f"  > Found {len(self.stop_shingles)} stop shingles (>{threshold_freq:.1%} freq) to prune.")

    def get_character_shingles(self, text, k=3):
        """
        Extracts the shingles from a corpus and removes the shingles that 
        qualify as stop shingles
        """
        cleaned = self.clean_text(text)
        if len(cleaned) < k:
            return {cleaned}
        
        shingles = {cleaned[i : i+k] for i in range(len(cleaned) - k + 1)}
        
        if self.stop_shingles:
            shingles = shingles - self.stop_shingles
            
        return shingles

    def get_model_words(self, product):
        """
        Extracts the model words from a corpus
        """
        model_words = set()

        # Check in the title for the corresponding regex
        title = product.get("title", "")
        matches = self.title_regex.findall(title)
        for m in matches:
            if isinstance(m, tuple):
                model_words.add(m[0].lower())
            else:
                model_words.add(m.lower())

        # Check in the feature map for the corresponding regex
        features = product.get("featuresMap", {})
        for key, value in features.items():
            kvp_matches = self.kvp_regex.findall(str(value))
            for m in kvp_matches:
                model_words.add(m.lower())

            if key.lower() == "brand":
                 model_words.add(str(value).lower())

        return model_words

    def get_hybrid_features(self, product, k=3, mw_weight=2):
        """
        Generates the hybrid feature set of shingles and model words
        """

        # Get shingles
        features = list(self.get_character_shingles(product.get('title', ""), k))
        
        model_tokens = self.get_model_words(product)
        product['model_words_cache'] = model_tokens
        
        # Add model words
        for token in model_tokens:
            is_alphanumeric = any(c.isalpha() for c in token) and any(c.isdigit() for c in token)
            
            if is_alphanumeric:
                for i in range(mw_weight):
                    features.append(f"MW_{token}_{i}")
            else:
                features.append(f"MW_{token}_0")
        
        return set(features)