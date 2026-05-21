from typing import List
import numpy as np

def length_bucket(query_len_tokens: int) -> str:
    if query_len_tokens <= 8:
        return "S"
    if query_len_tokens <= 20:
        return "M"
    if query_len_tokens <= 64:
        return "L"
    return "XL"


def noise_score_simple(text: str) -> float:
    upper = sum(ch.isupper() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    rare = sum(ch in {'/', '\\', '@', '#', '%'} for ch in text)

    return (upper + digits + 2 * rare) / max(1, len(text))


def nearest_clusters(
    query_vec: np.ndarray,
    centroids: np.ndarray,
    topC: int = 2
) -> List[int]:

    if centroids is None:
        return []

    q = query_vec / (np.linalg.norm(query_vec) + 1e-9)

    C = centroids / (
        np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
    )

    sims = (C @ q)

    idx = np.argsort(-sims)[:topC]

    return [int(i) for i in idx]