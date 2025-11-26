# text/search_engine.py

import math
import pickle
from collections import defaultdict

from text.preprocess import normalize
from text.index_builder import INDEX_PATH, DOC_NORMS_PATH, VOCAB_PATH
from backend.db import get_connection

with open(INDEX_PATH, "rb") as f:
    POSTINGS = pickle.load(f)

with open(DOC_NORMS_PATH, "rb") as f:
    DOC_NORMS = pickle.load(f)

with open(VOCAB_PATH, "rb") as f:
    VOCAB = pickle.load(f)


def search(query: str, k: int = 10):
    terms = normalize(query)

    # tf de query
    tf_q = defaultdict(int)
    for t in terms:
        tf_q[t] += 1

    # pesos tf-idf de la query
    w_q = {}
    for t, tf in tf_q.items():
        vocab_info = VOCAB.get(t)
        if not vocab_info:
            continue
        idf = vocab_info["idf"]
        w_q[t] = tf * idf

    # norma de la query
    norm_q = math.sqrt(sum(w * w for w in w_q.values()))
    if norm_q == 0:
        return []

    # acumulador doc_id -> numerador de coseno
    scores = defaultdict(float)

    for t, wqt in w_q.items():
        postings_t = POSTINGS.get(t)
        if not postings_t:
            continue
        for doc_id, wdt in postings_t.items():
            scores[doc_id] += wqt * wdt

    # convertir a coseno
    results = []
    for doc_id, num in scores.items():
        norm_d = DOC_NORMS.get(doc_id, 0.0)
        if norm_d == 0:
            continue
        score = num / (norm_q * norm_d)
        results.append((doc_id, score))

    # top-k
    results.sort(key=lambda x: x[1], reverse=True)
    results = results[:k]

    # traer metadata desde la BD
    conn = get_connection()
    cur = conn.cursor()
    doc_ids = [r[0] for r in results]
    cur.execute(
        "SELECT id, title, artist FROM documents WHERE id = ANY(%s)",
        (doc_ids,),
    )
    meta = {row[0]: row[1:] for row in cur.fetchall()}
    conn.close()

    final = []
    for doc_id, score in results:
        title, artist = meta.get(doc_id, ("", ""))
        final.append(
            {"id": doc_id, "title": title, "artist": artist, "score": score}
        )
    return final


if __name__ == "__main__":
    while True:
        q = input("consulta> ")
        if not q:
            break
        res = search(q, k=5)
        for r in res:
            print(f"{r['score']:.3f} - {r['title']} / {r['artist']}")
