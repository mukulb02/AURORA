from typing import List, Dict, Any
import numpy as np

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchValue
)

from app.config import (
    COLLECTION,
    MODEL_EMB,
    VECTOR_DIM,
    NUM_CLUSTERS,
    device
)

client = QdrantClient(url="http://localhost:6333")

embedder = SentenceTransformer(MODEL_EMB).to(device)

print("[ARS-LB] Loading dataset...")

ds = load_dataset(
    "mukulb/combined_medical_corpus",
    split="train"
)

corpus = ds["text"]
group_names = ds["group_name"]

print("[ARS-LB] Fitting TF-IDF...")

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

cluster_centroids = None
HAS_CLUSTER_FIELD = False


def qdrant_search_by_clusters(
    query_vector,
    clusters: List[int],
    limit: int
):

    if not clusters:
        return client.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=limit
        )

    results = []

    per = max(1, limit // len(clusters))

    for c in clusters:

        f = Filter(
            must=[
                FieldCondition(
                    key="cluster_id",
                    match=MatchValue(value=int(c))
                )
            ]
        )

        part = client.search(
            collection_name=COLLECTION,
            query_vector=query_vector,
            limit=per,
            query_filter=f
        )

        results.extend(part)

    return results[:limit]


def hybrid_retrieve(
    query: str,
    query_vec: np.ndarray,
    k_sparse: int,
    k_dense: int,
    clusters: List[int]
) -> List[Dict[str, Any]]:

    dense_hits = qdrant_search_by_clusters(
        query_vec,
        clusters,
        limit=k_dense
    )

    dense_docs = [{
        "text": r.payload["text"],
        "group": r.payload.get("group_name", "Unknown"),
        "cluster_id": r.payload.get("cluster_id", None),
        "score_dense": float(r.score),
        "score_sparse": 0.0
    } for r in dense_hits]

    tfidf_q = tfidf_vectorizer.transform([query])

    sparse_scores_arr = (
        tfidf_matrix.dot(tfidf_q.T)
        .toarray()
        .flatten()
    )

    idxs = np.argsort(sparse_scores_arr)[::-1][:k_sparse]

    sparse_docs = [{
        "text": corpus[i],
        "group": "Unknown",
        "cluster_id": None,
        "score_dense": 0.0,
        "score_sparse": float(sparse_scores_arr[i])
    } for i in idxs]

    return dense_docs + sparse_docs