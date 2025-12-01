import pickle
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  
HIST_PATH = ROOT / "indexes" / "histograms.pkl"
INV_PATH = ROOT / "indexes" / "inverted_index.pkl"



def load_all():
    with HIST_PATH.open("rb") as f:
        histograms = pickle.load(f)
    with INV_PATH.open("rb") as f:
        inverted = pickle.load(f)
    return histograms, inverted


def knn_inverted(query_img: str, k: int = 8):
    histograms, inverted = load_all()

    if query_img not in histograms:
        raise KeyError("Query no existe en histogramas")

    qhist = histograms[query_img]

    # palabras visuales relevantes para la query
    q_words = np.argsort(qhist)[-10:]  # top 10 visual words

    candidates = set()
    for w in q_words:
        if w in inverted:
            candidates.update(inverted[w])

    candidates.discard(query_img)

    results = []
    for img in candidates:
        sim = float(np.dot(qhist, histograms[img]))
        results.append((sim, img))

    results.sort(reverse=True, key=lambda x: x[0])
    return results[:k]


def cli():
    h, _ = load_all()
    print("[INFO] Imagenes disponibles:")
    for name in list(h.keys())[:10]:
        print(" -", name)
    print()

    while True:
        q = input("query image> ").strip()
        if not q:
            break
        try:
            topk = knn_inverted(q)
            print("\nTop-k similares:")
            for s, name in topk:
                print(f" {s:.3f} - {name}")
            print()
        except Exception as e:
            print("[ERROR]", e)


if __name__ == "__main__":
    cli()