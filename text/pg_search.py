# text/pg_search.py

import time
from backend.db import get_connection


def pg_search(query: str, k: int = 5):
    conn = get_connection()
    cur = conn.cursor()

    # usamos plainto_tsquery para algo parecido a lenguaje natural
    sql = """
    SELECT id, title, artist, lyric,
           ts_rank(ts, plainto_tsquery('english', %s)) AS rank
    FROM documents
    WHERE ts @@ plainto_tsquery('english', %s)
    ORDER BY rank DESC
    LIMIT %s;
    """

    start = time.time()
    cur.execute(sql, (query, query, k))
    rows = cur.fetchall()
    elapsed = time.time() - start

    cur.close()
    conn.close()

    results = [
        {"id": r[0], "title": r[1], "artist": r[2], "lyric": r[3], "score": float(r[4])}
        for r in rows
    ]
    return results, elapsed


if __name__ == "__main__":
    while True:
        q = input("consulta (Postgres)> ")
        if not q:
            break
        res, t = pg_search(q, k=5)
        print(f"Tiempo PostgreSQL: {t:.4f} s")
        for r in res:
            print(f"{r['score']:.3f} - {r['title']} / {r['artist']}")
