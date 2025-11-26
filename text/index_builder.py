# text/index_builder.py

import math
import pickle
from collections import defaultdict

from backend.db import get_connection
from text.preprocess import normalize

INDEX_PATH = "data/text/index.pkl"
DOC_NORMS_PATH = "data/text/doc_norms.pkl"
VOCAB_PATH = "data/text/vocab.pkl"


def build_index():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, title, lyric FROM documents")
    rows = cur.fetchall()
    conn.close()

    N = len(rows)
    print(f"Construyendo indice para {N} documentos...")

    # postings[term] = dict(doc_id -> tf)
    postings = defaultdict(lambda: defaultdict(int))

    for doc_id, title, lyric in rows:
        text = (title or "") + " " + (lyric or "")
        terms = normalize(text)
        for t in terms:
            postings[t][doc_id] += 1

    # Calcular IDF y pesos tf-idf
    vocab = {}
    doc_norms = defaultdict(float)

    for term_idx, (term, docs) in enumerate(postings.items()):
        df = len(docs)
        if df == 0:
            continue
        idf = math.log(N / df)

        vocab[term] = {
            "idf": idf,
            "df": df,
        }

        for doc_id, tf in docs.items():
            w = tf * idf
            postings[term][doc_id] = w
            doc_norms[doc_id] += w * w

    # norma final
    for doc_id in doc_norms:
        doc_norms[doc_id] = math.sqrt(doc_norms[doc_id])

    # guardar en disco
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(postings, f)

    with open(DOC_NORMS_PATH, "wb") as f:
        pickle.dump(dict(doc_norms), f)

    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)

    print("Indice construido y guardado en disco.")


if __name__ == "__main__":
    build_index()
