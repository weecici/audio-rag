import math


def calc_idf(df: int, N: int) -> float:
    inner = (N - df + 0.5) / (df + 0.5)
    return math.log(inner)


def calc_tfidf(tf: int, idf: float) -> float:
    return tf * idf


def calc_okapi_bm25(
    tf: int, idf: float, dl: int, avg_dl: float, k1: float = 1.5, b: float = 0.75
) -> float:
    numerator = tf
    denominator = tf + k1 * (1 - b + b * dl / avg_dl)
    return idf * (numerator / denominator)
