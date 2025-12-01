from flask import Flask, render_template, request
import time
import os
import sys 
import pdb
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from text.search_engine import search as myindex_search
from text.pg_search import pg_search


from multimedia.knn_sequential import knn_query as knn_seq
from multimedia.knn_inverted import knn_inverted as knn_inv

from pathlib import Path


app = Flask(__name__)

ROOT = Path(__file__).resolve().parents[1]  

IMG_DIR = ROOT / "data"  / "images"


# BUSQUEDA DE TEXTO
@app.route("/", methods=["GET", "POST"])
def text_search():
    results_myindex = None
    results_pg = None
    query = ""
    k = 5
    time_my = None
    time_pg = None

    if request.method == "POST":
        query = request.form.get("query")
        k = int(request.form.get("k"))

        # --- MyIndex ---
        t0 = time.time()
        results_myindex = myindex_search(query, k=k)
        if results_myindex:
            for res in results_myindex:
                res["lyric"] = res["lyric"][0:100]
        time_my = round(time.time() - t0, 4)

        # --- PostgreSQL ---
        t0 = time.time()
        results_pg = pg_search(query, k=k)
        if results_pg:
            results_pg = results_pg[0]
            for res in results_pg:
                res["lyric"] = res["lyric"][0:100]
        time_pg = round(time.time() - t0, 4)

        #pdb.set_trace()

    return render_template(
        "text_search.html",
        results_myindex=results_myindex,
        results_pg=results_pg,
        query=query,
        k=k,
        time_my=time_my,
        time_pg=time_pg
    )


# BÚSQUEDA DE IMÁGENES
@app.route("/images", methods=["GET", "POST"])
def image_search():
    files = os.listdir(IMG_DIR)
    files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    files = files [1:50000]
    #pdb.set_trace()
    query_img = None
    k = 5
    seq_results = None
    seq_time = None

    inv_results = None
    inv_time = None

    if request.method == "POST":
        query_img = request.form.get("image")
        k = int(request.form.get("k"))

        # KNN secuencial
        t0 = time.time()
        seq_results = knn_seq(query_img, k=k)
        seq_time = round(time.time() - t0, 4)

        # KNN invertido
        t0 = time.time()
        inv_results = knn_inv(query_img, k=k)
        inv_time = round(time.time() - t0, 4)

    return render_template(
        "image_search.html",
        images=files,
        query_img=query_img,
        k=k,
        seq_results=seq_results,
        seq_time=seq_time,
        inv_results=inv_results,
        inv_time=inv_time
    )


if __name__ == "__main__":
    app.run(debug=True)