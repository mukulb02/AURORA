# AURORA

Adaptive Unified Retrieval with Optimized Re-Ranking and Reliability.

## Features

- Hybrid Dense + Sparse Retrieval
- Cluster-Aware Qdrant Routing
- Cross-Encoder Reranking
- Self-Calibrating Confidence Margins
- Adaptive Retrieval Depth
- FastAPI Deployment

## Architecture

Query
→ Length Bucket Routing
→ Hybrid Retrieval
→ Cluster Routing
→ Cross-Encoder Reranking
→ Confidence Calibration
→ Generation

## Installation

```bash
git clone https://github.com/mukulb02/AURORA.git
cd ARS-LB

pip install -r requirements.txt
```

## Run

```bash
uvicorn app.main:app --reload
```

## Citation

```bibtex
@misc{arslb2026,
  author       = {Mukul Sharma},
  title        = {ARS-LB: Adaptive Retrieval Strategy with Length Buckets},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/mukulb02/AURORA}}
}
```
