import pickle
from pathlib import Path
import heapq
import numpy as np
#
#from utils import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]  
HIST_PATH = ROOT / "indexes" / "histograms.pkl"

K = 8  


def load_histograms():
    with HIST_PATH.open("rb") as f:
        return pickle.load(f)  


def knn_query(query_img,histograms, k=K):
   # histograms = load_histograms()
    if query_img not in histograms:
        raise KeyError(f"{query_img} no existe en los histogramas")

    q = histograms[query_img]
    scores = []

    for img_name, h in histograms.items():
        if img_name == query_img:
            continue
        s = float(np.dot(q, h))  #coseno porque estan normalizados
        scores.append((s, img_name))

    topk = heapq.nlargest(k, scores, key=lambda x: x[0])
    return topk


def cli():
    histograms = load_histograms()
    print(f"[INFO] {len(histograms)} imagenes cargadas.")
    print("Escribe el nombre de una imagen (por ejemplo: 10000.jpg). Enter vacio para salir.\n")

    while True:
        q = input("query image> ").strip()
        if not q:
            break

        try:
            results = knn_query(q, histograms)
        except KeyError:
            print(f"[WARN] No encuentro {q} en los histogramas.")
            continue

        print(f"Top-{K} similares a {q}:")
        for score, img_name in results:
            print(f"  {score:.3f}  -  {img_name}")
        print()


if __name__ == "__main__":
    cli()