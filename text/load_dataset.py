# text/load_dataset.py

import csv
import os
from pathlib import Path

import psycopg2

from backend.config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASS,
)

DATA_PATH = Path("data/text/spotify_songs.csv")


def get_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASS,
    )


def load_lyrics():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No encuentro el archivo {DATA_PATH}")

    conn = get_connection()
    cur = conn.cursor()

    with DATA_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            title = row.get("track_name", "")
            artist = row.get("track_artist", "")
            lyric = row.get("lyrics", "")

            cur.execute(
                """
                INSERT INTO documents (title, artist, lyric)
                VALUES (%s, %s, %s)
                """,
                (title, artist, lyric),
            )
            count += 1
            if count % 100 == 0:
                conn.commit()
                print(f"Insertados {count} documentos...")

    conn.commit()
    print(f"Listo, insertados {count} documentos.")
    cur.close()
    conn.close()


if __name__ == "__main__":
    load_lyrics()
