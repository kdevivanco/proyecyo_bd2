# multimedia/utils.py

import pickle
import numpy as np
from pathlib import Path


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def save_pickle(obj, path):
    path = Path(path)
    ensure_parent_dir(path)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)


def cosine_similarity(a, b):
    """Similitud coseno entre dos vectores 1D."""
    a = np.asarray(a)
    b = np.asarray(b)
    num = float(np.dot(a, b))
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return num / den
