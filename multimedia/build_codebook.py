
import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  

RAW_FEATS = ROOT / "indexes" / "features_raw.pkl"
KMEANS_PATH = ROOT / "indexes" / "kmeans_model.pk1"
CODEBOOK_PATH = ROOT / "indexes" / "codebook.pkl"

#RAW_FEATS = Path("indexes/features_raw.pkl")
#CODEBOOK_PATH = Path("indexes/codebook.pkl")
#KMEANS_PATH = Path("indexes/kmeans_model.pkl")

K = 100   # número de visual words 


def build_codebook():
    print("[INFO] Cargando descriptores...")
    with RAW_FEATS.open("rb") as f:
        feats = pickle.load(f)

    all_desc = []

    for img, desc in feats.items():
        if desc is not None and len(desc) > 0:
            all_desc.append(desc)

    all_desc = np.vstack(all_desc)
    print(f"[INFO] Descriptores totales para KMeans: {all_desc.shape}")

    print("[INFO] Ejecutando KMeans...")
    kmeans = MiniBatchKMeans(
        n_clusters=K,
        random_state=42,
        batch_size=2048
    ).fit(all_desc)

    print("[INFO] Guardando codebook...")
    CODEBOOK_PATH.parent.mkdir(exist_ok=True)

    with CODEBOOK_PATH.open("wb") as f:
        pickle.dump(kmeans.cluster_centers_, f)

    with KMEANS_PATH.open("wb") as f:
        pickle.dump(kmeans, f)

    print("[OK] Codebook generado.")


if __name__ == "__main__":
    build_codebook()
