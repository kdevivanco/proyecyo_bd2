import time
import random
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HIST_PATH = ROOT / "indexes" / "histograms.pkl"

NS = [1000, 2000, 4000, 8000, 16000, 32000, 44000]
K = 8  # K del top-k


def load_histograms() -> Dict[str, np.ndarray]:
    with HIST_PATH.open("rb") as f:
        histograms = pickle.load(f)
    return histograms


def build_inverted_index(histograms: Dict[str, np.ndarray]):
    inverted = defaultdict(list)
    for img_name, hist in histograms.items():
        for idx, val in enumerate(hist):
            if val > 0.0:
                inverted[idx].append(img_name)
    return inverted


def knn_sequential_local(
    query_img: str,
    histograms: Dict[str, np.ndarray],
    k: int = K,
) -> List[Tuple[float, str]]:
   
    if query_img not in histograms:
        raise KeyError(f"{query_img} no existe en los histogramas")

    q = histograms[query_img]
    scores: List[Tuple[float, str]] = []

    for img_name, h in histograms.items():
        if img_name == query_img:
            continue
        s = float(np.dot(q, h))  # coseno porque estan normalizados
        scores.append((s, img_name))

    # top-k
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:k]


def knn_inverted_local(
    query_img: str,
    histograms: Dict[str, np.ndarray],
    inverted,
    k: int = K,
    top_words: int = 10,
) -> List[Tuple[float, str]]:
  
    if query_img not in histograms:
        raise KeyError("Query no existe en histogramas")

    qhist = histograms[query_img]

    q_words = np.argsort(qhist)[-top_words:]

    candidates = set()
    for w in q_words:
        if w in inverted:
            candidates.update(inverted[w])

    candidates.discard(query_img)

    results: List[Tuple[float, str]] = []
    for img in candidates:
        sim = float(np.dot(qhist, histograms[img]))
        results.append((sim, img))

    results.sort(reverse=True, key=lambda x: x[0])
    return results[:k]


def bench_for_N(
    full_histograms: Dict[str, np.ndarray],
    N: int,
    num_queries: int = 5,
    k: int = K,
):
   
    img_ids = list(full_histograms.keys())
    if N > len(img_ids):
        N = len(img_ids)

    subset_ids = img_ids[:N]
    subset_hists = {img_id: full_histograms[img_id] for img_id in subset_ids}

    inverted = build_inverted_index(subset_hists)

    query_ids = random.sample(subset_ids, min(num_queries, len(subset_ids)))

    # --- KNN SECUENCIAL ---
    t0 = time.time()
    for qid in query_ids:
        _ = knn_sequential_local(qid, subset_hists, k=k)
    t_seq = (time.time() - t0) / len(query_ids)

    # --- KNN INVERTIDO ---
    t0 = time.time()
    for qid in query_ids:
        _ = knn_inverted_local(qid, subset_hists, inverted, k=k)
    t_inv = (time.time() - t0) / len(query_ids)

    return t_seq, t_inv


def main():
    histograms = load_histograms()
    print(f"Total imagenes disponibles: {len(histograms)}\n")

    random.seed(42)  #

    for N in NS:
        t_seq, t_inv = bench_for_N(histograms, N)
        print(
            f"N = {N:6d}  |  KNN secuencial: {t_seq:.4f} s  |  KNN invertido: {t_inv:.4f} s"
        )


if __name__ == "__main__":
    main()