import os
import random
import tempfile
from pathlib import Path
import urllib.request

import numpy as np
import pytest

# Keep hash-based nondeterminism stable where possible
os.environ.setdefault("PYTHONHASHSEED", "0")


@pytest.fixture(autouse=True)
def _seed_everything():
    random.seed(0)
    np.random.seed(0)
    try:
        import tensorflow as tf
        try:
            tf.keras.utils.set_random_seed(0)
        except Exception:
            tf.random.set_seed(0)
    except Exception:
        pass
    yield


@pytest.fixture
def change_test_dir(tmp_path, monkeypatch):
    """Run a test in an isolated temporary working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(scope="session")
def test_w2v_model_path(tmp_path_factory):
    """Create a small deterministic word2vec-format model for tests (no network)."""
    from gensim.models import KeyedVectors

    path = tmp_path_factory.mktemp("models") / "test_w2v_model.bin"
    tokens = [
        "president", "speaks", "talks", "fan", "book", "apple", "orange",
        "artificial", "intelligence", "linear", "algebra", "calculus",
        "quantum", "mechanics", "physics", "theology", "god", "faith",
        "learning", "eschatology", "stem", "cell", "grant", "system", "biology",
        "mathematics", "topology", "church", "bible", "research", "science",
        "software", "computer", "network", "model"
    ]
    dim = 8
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(len(tokens), dim)).astype(np.float32)
    # inject simple structure so related words are closer and WMD expectations are stable
    def set_vec(tok, base):
        vectors[tokens.index(tok)] = base
    v_pres = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    v_speak = np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    v_talk = np.array([0.88, 0.12, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    v_fan = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    v_book = np.array([0, 0.9, 0.1, 0, 0, 0, 0, 0], dtype=np.float32)
    v_apple = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    v_orange = np.array([0, 0, 0.95, 0.05, 0, 0, 0, 0], dtype=np.float32)
    for tok, vec in [
        ("president", v_pres), ("speaks", v_speak), ("talks", v_talk),
        ("fan", v_fan), ("book", v_book), ("apple", v_apple), ("orange", v_orange),
        ("mathematics", np.array([0, 0, 0, 1, 0.2, 0, 0, 0], np.float32)),
        ("physics", np.array([0, 0, 0, 0.8, 0.4, 0, 0, 0], np.float32)),
        ("theology", np.array([0, 0, 0, 0, 0, 1, 0.2, 0], np.float32)),
        ("linear", np.array([0, 0, 0, 0.9, 0.1, 0, 0, 0], np.float32)),
        ("algebra", np.array([0, 0, 0, 0.95, 0.05, 0, 0, 0], np.float32)),
        ("calculus", np.array([0, 0, 0, 0.92, 0.08, 0, 0, 0], np.float32)),
        ("topology", np.array([0, 0, 0, 0.93, 0.07, 0, 0, 0], np.float32)),
        ("quantum", np.array([0, 0, 0, 0.7, 0.5, 0, 0, 0], np.float32)),
        ("mechanics", np.array([0, 0, 0, 0.75, 0.45, 0, 0, 0], np.float32)),
        ("god", np.array([0, 0, 0, 0, 0, 0.95, 0.15, 0], np.float32)),
        ("faith", np.array([0, 0, 0, 0, 0, 0.9, 0.2, 0], np.float32)),
        ("church", np.array([0, 0, 0, 0, 0, 0.92, 0.18, 0], np.float32)),
        ("bible", np.array([0, 0, 0, 0, 0, 0.91, 0.17, 0], np.float32)),
        ("eschatology", np.array([0, 0, 0, 0, 0, 0.97, 0.1, 0], np.float32)),
        ("artificial", np.array([0, 0, 0, 0.6, 0.1, 0, 0.3, 0], np.float32)),
        ("intelligence", np.array([0, 0, 0, 0.55, 0.15, 0, 0.35, 0], np.float32)),
        ("learning", np.array([0, 0, 0, 0.58, 0.12, 0, 0.32, 0], np.float32)),
        ("computer", np.array([0, 0, 0, 0.5, 0.1, 0, 0.4, 0], np.float32)),
        ("software", np.array([0, 0, 0, 0.48, 0.12, 0, 0.42, 0], np.float32)),
        ("science", np.array([0, 0, 0, 0.7, 0.3, 0, 0.1, 0], np.float32)),
        ("research", np.array([0, 0, 0, 0.65, 0.25, 0, 0.15, 0], np.float32)),
        ("stem", np.array([0, 0, 0, 0.5, 0.35, 0, 0.1, 0.05], np.float32)),
        ("cell", np.array([0, 0, 0, 0.45, 0.4, 0, 0.1, 0.05], np.float32)),
        ("biology", np.array([0, 0, 0, 0.5, 0.38, 0, 0.08, 0.04], np.float32)),
        ("system", np.array([0, 0, 0, 0.55, 0.2, 0, 0.2, 0.05], np.float32)),
        ("grant", np.array([0, 0, 0, 0.4, 0.2, 0.1, 0.1, 0.2], np.float32)),
        ("network", np.array([0, 0, 0, 0.52, 0.18, 0, 0.25, 0.05], np.float32)),
        ("model", np.array([0, 0, 0, 0.57, 0.2, 0, 0.2, 0.03], np.float32)),
    ]:
        set_vec(tok, vec)
    # normalize
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    kv = KeyedVectors(vector_size=dim)
    kv.add_vectors(tokens, vectors)
    kv.save_word2vec_format(str(path), binary=True)
    return path


@pytest.fixture(scope="session")
def big_text():
    """Small deterministic corpus for Norvig/char-encoder tests (no network)."""
    return (
        "the quick brown fox jumps over the lazy dog\n"
        "apple apply application applied apprentice appeal\n"
        "natural language processing and text mining are useful\n"
        "spelling correction learns from frequent words in a corpus\n"
        "this is a small sample text for testing purposes\n"
    ) * 5
