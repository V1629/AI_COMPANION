import numpy as np
from ai_companion.embeddings import CharNGramHasher

def test_embedding_consistency():
    e = CharNGramHasher(dim=64, ngram_min=3, ngram_max=4)
    a = e.embed("I'm fine")
    b = e.embed("I'm fine")
    assert a.shape == (64,)
    np.testing.assert_allclose(a, b, atol=1e-6)

def test_embedding_different_for_different_texts():
    e = CharNGramHasher(dim=64, ngram_min=3, ngram_max=4)
    a = e.embed("I am fine")
    b = e.embed("I am not fine")
    # Ensure embeddings are not identical
    assert not np.allclose(a, b)