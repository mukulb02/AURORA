import torch
import numpy as np

# -------------------------
# Reproducibility
# -------------------------
np.random.seed(42)
torch.manual_seed(42)

# -------------------------
# Runtime Mode
# -------------------------
EVAL_MODE = True

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Models
# -------------------------
COLLECTION = "combined_medical_corpus"

MODEL_EMB = "aleynahukmet/bge-medical-small-en-v1.5"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEN_MODEL = "facebook/bart-large-cnn"

# -------------------------
# Retrieval Parameters
# -------------------------
VECTOR_DIM = 384
NUM_CLUSTERS = 32
ROLLING_WINDOW = 200

CONF_MARGIN_DEFAULT = 0.20
ESCALATION_MULTIPLIER = 1.5

INITIAL_K = {
    "S": (30, 20),
    "M": (25, 20),
    "L": (20, 30),
    "XL": (25, 20),
}

MAX_RERANK_M = {
    "S": 50,
    "M": 40,
    "L": 50,
    "XL": 40,
}