"""
Custom lightweight embeddings based on character n-gram hashing.
No pretrained models; deterministic and tunable dimensionality.
"""
import numpy as np
import hashlib
from typing import List


class CharNGramHasher:
    """
    Create fixed-size embeddings by hashing character n-grams into vector bins.
    This is fast, incremental, and privacy-friendly (no external model).
    """

    def __init__(self, dim: int = 128, ngram_min: int = 3, ngram_max: int = 5):
        self.dim = dim
        self.ngram_min = ngram_min
        self.ngram_max = ngram_max

    def _ngrams(self, text: str) -> List[str]:
        s = f" {text.lower()} "
        grams = []
        for n in range(self.ngram_min, self.ngram_max + 1):
            for i in range(len(s) - n + 1):
                grams.append(s[i : i + n])
        return grams

    def _hash_to_index(self, gram: str) -> int:
        h = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "big") % self.dim

    def embed(self, text: str) -> np.ndarray:
        """
        Return a normalized embedding vector for input text.
        """
        vec = np.zeros(self.dim, dtype=float)
        grams = self._ngrams(text)
        for g in grams:
            idx = self._hash_to_index(g)
            vec[idx] += 1.0
        # apply sqrt weighting and normalize
        vec = np.sqrt(vec)
        norm = np.linalg.norm(vec) + 1e-9
        return vec / norm

    def batch_embed(self, texts: List[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts], axis=0)