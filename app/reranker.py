from typing import List, Dict, Any, Tuple
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from app.config import CE_MODEL, device

cross_encoder_tokenizer = AutoTokenizer.from_pretrained(CE_MODEL)

cross_encoder_model = (
    AutoModelForSequenceClassification
    .from_pretrained(CE_MODEL)
    .to(device)
)


def rerank_and_margin(
    query: str,
    docs: List[Dict[str, Any]],
    top_m: int
) -> Tuple[List[Dict[str, Any]], float]:

    if not docs:
        return [], 0.0

    pairs = [(query, d["text"]) for d in docs]

    inputs = cross_encoder_tokenizer(
        pairs,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = (
            cross_encoder_model(**inputs)
            .logits
            .squeeze(-1)
            .detach()
            .cpu()
            .numpy()
        )

    mean, std = logits.mean(), logits.std() + 1e-6
    logits_norm = (logits - mean) / std

    for d, s in zip(docs, logits_norm):
        d["rerank"] = float(s)

    w_dense, w_sparse, w_rerank = 0.15, 0.05, 0.80

    for d in docs:
        d["final"] = (
            w_dense * d.get("score_dense", 0.0)
            + w_sparse * d.get("score_sparse", 0.0)
            + w_rerank * d["rerank"]
        )

    order = np.argsort([-d["final"] for d in docs])

    docs_sorted = [
        docs[i]
        for i in order[:max(top_m, 2)]
    ]

    if len(docs_sorted) >= 2:
        margin = (
            docs_sorted[0]["rerank"]
            - docs_sorted[1]["rerank"]
        )
    else:
        margin = 1e9

    return docs_sorted, float(margin)