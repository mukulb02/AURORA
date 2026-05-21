from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import time

from app.routing import (
    length_bucket,
    noise_score_simple,
    nearest_clusters
)

from app.retrieval import (
    embedder,
    cluster_centroids,
    hybrid_retrieve
)

from app.reranker import rerank_and_margin

from app.calibration import (
    depth_manager,
    calibrator
)

from app.generation import (
    truncate_context,
    hf_generate
)

from app.config import (
    EVAL_MODE,
    MAX_RERANK_M,
    ESCALATION_MULTIPLIER
)

app = FastAPI()


class Query(BaseModel):
    query: str


def ars_lb_search(query: str, limit: int = 5):

    qvec = embedder.encode(query)

    qlen = len(query.split())

    bucket = length_bucket(qlen)

    noise = noise_score_simple(query)

    kS, kD = depth_manager.suggest(bucket)

    rerank_m = MAX_RERANK_M[bucket]

    clusters = nearest_clusters(
        qvec,
        cluster_centroids,
        topC=2
    )

    docs = hybrid_retrieve(
        query,
        qvec,
        kS,
        kD,
        clusters
    )

    ranked, margin = rerank_and_margin(
        query,
        docs,
        top_m=max(rerank_m, limit)
    )

    delta = calibrator.delta()

    calibrator.record(margin)

    return ranked[:limit], bucket, noise, margin, delta


def chat(query: str):

    t0 = time.time()

    top_docs, bucket, noise, margin, delta = (
        ars_lb_search(query)
    )

    ctx = " ".join([
        d["text"]
        for d in top_docs[:3]
    ])

    ctx = truncate_context(ctx)

    if EVAL_MODE or margin >= delta:
        answer = hf_generate(ctx, query)
    else:
        answer = (
            "I am not sufficiently confident "
            "in the retrieved evidence."
        )

    return {
        "answer": answer,
        "metadata": {
            "bucket": bucket,
            "noise": noise,
            "margin": margin,
            "delta_used": delta
        },
        "processing_time": f"{time.time()-t0:.2f}s"
    }


@app.get("/predict")
def predict(query: str):

    try:
        return chat(query)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )