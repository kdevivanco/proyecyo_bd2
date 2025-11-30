
import pickle
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]  
HIST_PATH = ROOT / "indexes" / "histograms.pkl"
INV_PATH = ROOT / "indexes" / "inverted_index.pkl"


def build_inverted_index():
    print("[INFO] Cargando histogramas…")
    with HIST_PATH.open("rb") as f:
        histograms = pickle.load(f)

    inverted = defaultdict(list)  

    print("[INFO] Construyendo indice invertido…")
    for img_name, hist in histograms.items():
        # palabras visuales no cero
        words = hist.nonzero()[0]
        for w in words:
            inverted[int(w)].append(img_name)

    print(f"[INFO] Indice invertido construido con {len(inverted)} visual words.")
    with INV_PATH.open("wb") as f:
        pickle.dump(dict(inverted), f)

    print(f"[OK] Guardado en {INV_PATH}")


if __name__ == "__main__":
    build_inverted_index()