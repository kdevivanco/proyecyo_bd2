# text/preprocess.py

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

_ps = PorterStemmer()
_stopwords = None


def init_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")


def normalize(text: str):
    global _stopwords
    if _stopwords is None:
        _stopwords = set(stopwords.words("english"))

    text = text.lower()
    tokens = re.findall(r"\w+", text)

    result = []
    for tok in tokens:
        if tok in _stopwords:
            continue
        stem = _ps.stem(tok)
        result.append(stem)
    return result
