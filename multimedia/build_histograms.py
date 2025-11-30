import pickle
import numpy as np
from pathlib import Path

from utils import save_pickle

ROOT = Path(__file__).resolve().parents[1]  

RAW_FEATS = ROOT / "indexes" / "features_raw.pkl"
KMEANS_PATH = ROOT / "indexes" / "kmeans_model.pk1"
HIST_PATH = ROOT / "indexes" / "histograms.pkl"



def build_histograms():
    print("[INFO] Cargando descriptores crudos...")
    with RAW_FEATS.open("rb") as f:
        feats = pickle.load(f)  

    print("[INFO] Cargando modelo KMeans...")
    with KMEANS_PATH.open("rb") as f:
        kmeans = pickle.load(f)

    K = kmeans.n_clusters
    histograms = {}  

    for img_name, desc in feats.items():
        if desc is None or len(desc) == 0:
            continue

        labels = kmeans.predict(desc) 
        hist = np.bincount(labels, minlength=K).astype(np.float32)

        norm = np.linalg.norm(hist) + 1e-8
        hist = hist / norm

        histograms[img_name] = hist

    save_pickle(histograms, HIST_PATH)
    print(f"[OK] Histogramas guardados en {HIST_PATH} ({len(histograms)} imagenes)")


if __name__ == "__main__":
    build_histograms()