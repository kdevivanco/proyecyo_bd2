import time
from knn_sequential import knn_query, load_histograms
from knn_inverted import knn_inverted

h = load_histograms()
img = list(h.keys())[0]

t0 = time.time()
knn_query(img, h)
print("Secuencial:", time.time() - t0)

t0 = time.time()
knn_inverted(img)
print("Inverted:", time.time() - t0)