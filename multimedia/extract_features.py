# multimedia/extract_features.py

import cv2
import numpy as np
from pathlib import Path

from utils import save_pickle

ROOT = Path(__file__).resolve().parents[1]  
DATA_PATH = ROOT / "data" / "images"
OUT_PATH = ROOT / "indexes" / "features_raw.pkl"

def get_image_files():
    """Devuelve la lista de imágenes (.jpg/.jpeg/.png) en DATA_PATH."""
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in DATA_PATH.iterdir() if p.suffix.lower() in exts]


def extract_sift_features(img_path: Path):
    """
    Extrae descriptores SIFT de una imagen.
    Devuelve una matriz (N x 128) o None si no se pudo leer.
    """
    # leemos en gris (SIFT trabaja en escala de grises)
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARN] No pude leer {img_path}")
        return None

    # SIFT
    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        # fallback por si está en el módulo xfeatures2d
        sift = cv2.xfeatures2d.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(img, None)

    if descriptors is None:
        print(f"[INFO] {img_path.name}: 0 descriptores")
        return None

    print(f"[INFO] {img_path.name}: {len(descriptors)} descriptores")
    return descriptors.astype(np.float32)


def extract_all_features():
    """
    Recorre todas las imágenes y extrae sus descriptores SIFT.
    Guarda un dict: { "archivo.jpg": matriz_Nx128 } en OUT_PATH.
    """
    files = get_image_files()
    if not files:
        raise FileNotFoundError(
            f"No encontré imágenes en {DATA_PATH}. "
            "Copia algunas .jpg/.png ahí."
        )

    features = {}

    for img_file in files:
        desc = extract_sift_features(img_file)
        if desc is not None:
            features[img_file.name] = desc

    if not features:
        raise RuntimeError("No se obtuvieron descriptores de ninguna imagen.")

    save_pickle(features, OUT_PATH)
    print(f"[OK] Features extraídos de {len(features)} imágenes.")
    print(f"[OK] Guardado en: {OUT_PATH}")


if __name__ == "__main__":
    extract_all_features()
