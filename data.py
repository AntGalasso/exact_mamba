"""
data.py — Data loading, batching, and feature extraction

Responsibilities:
- Load real dataset (WikiText-2 via HuggingFace) OR fallback synthetic corpus
- Tokenize with GPT-2 tokenizer
- Produce deterministic batches
- Extract (X, Y) for linear solver pipeline
"""

import os
import pickle
import numpy as np

# Optional dependencies
try:
    import torch
except ImportError:
    torch = None

try:
    from datasets import load_dataset
    import tiktoken
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from extract import extract_XY, verify_dimensions
from config import Config

CACHE_PATH = ".cache/corpus.pkl"


# ============================================================
# 1. LOAD CORPUS
# ============================================================

def load_wikitext2(cfg):
    """
    Returns:
        train_ids: array of token ids
        val_ids:   array of token ids
    """

    if os.path.exists(CACHE_PATH):
        print("  Loaded from cache (.cache/corpus.pkl)")
        with open(CACHE_PATH, "rb") as f:
            data = pickle.load(f)
        return data["train"], data["val"]

    if HF_AVAILABLE:
        print("  Downloading WikiText-2 from HuggingFace...")

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

        print("  Tokenising with GPT-2 tokenizer...")
        enc = tiktoken.get_encoding("gpt2")

        def tokenize(split):
            text = "\n\n".join(dataset[split]["text"])
            ids = enc.encode(text)
            return np.array(ids, dtype=np.int64)

        train_ids = tokenize("train")
        val_ids   = tokenize("validation")

    else:
        print("  ⚠️ HuggingFace not available — using synthetic Zipf corpus")

        train_ids = _generate_zipf_corpus(cfg, size=2_000_000)
        val_ids   = _generate_zipf_corpus(cfg, size=200_000)

    # Save cache
    os.makedirs(".cache", exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"train": train_ids, "val": val_ids}, f)

    print("  Saved to .cache/corpus.pkl")

    return train_ids, val_ids


# ============================================================
# 2. SYNTHETIC DATA (fallback)
# ============================================================

def _generate_zipf_corpus(cfg, size):
    """
    Generate synthetic token distribution ~ Zipf law
    """
    vocab_size = cfg.vocab_size

    ranks = np.arange(1, vocab_size + 1)
    probs = 1.0 / ranks
    probs /= probs.sum()

    tokens = np.random.choice(vocab_size, size=size, p=probs)

    return tokens.astype(np.int64)


# ============================================================
# 3. FIXED BATCH (DETERMINISTIC)
# ============================================================

def get_fixed_batch(train_ids, cfg):
    """
    Deterministic batch used for experiments (X, Y extraction)

    Returns:
        input_ids:  [B, T]
        target_ids: [B, T]
    """

    B = cfg.batch_size
    T = cfg.seq_len

    start = 0
    x = train_ids[start : start + B * T + 1]

    input_ids  = x[:-1].reshape(B, T)
    target_ids = x[1:].reshape(B, T)

    return _to_tensor_if_possible(input_ids, target_ids)


# ============================================================
# 4. SEQUENTIAL BATCHES (for training)
# ============================================================

def get_sequential_batches(train_ids, cfg, max_batches=None):
    """
    Generate sequential batches for training

    Returns:
        list of (input_ids, target_ids)
    """

    B = cfg.batch_size
    T = cfg.seq_len

    total_tokens = len(train_ids)
    batches = []

    n_batches = (total_tokens - 1) // (B * T)

    if max_batches is not None:
        n_batches = min(n_batches, max_batches)

    for i in range(n_batches):
        start = i * B * T
        chunk = train_ids[start : start + B * T + 1]

        if len(chunk) < B * T + 1:
            break

        input_ids  = chunk[:-1].reshape(B, T)
        target_ids = chunk[1:].reshape(B, T)

        batches.append(_to_tensor_if_possible(input_ids, target_ids))

    return batches


# ============================================================
# 5. BUILD (X, Y) DATASET — 🔥 CORE STEP 1
# ============================================================

def build_dataset(model, input_ids, target_ids, layer_idx, cfg: Config):
    """
    Extract X, Y from model and prepare for solvers.

    Returns:
        dict with:
            X, Y
            OR
            X_train, Y_train, X_val, Y_val
    """

    # -----------------------------
    # Shape checks
    # -----------------------------
    if isinstance(input_ids, torch.Tensor):
        B, T = input_ids.shape
    else:
        B, T = input_ids.shape

    assert B == cfg.batch_size, f"Batch mismatch: {B} vs {cfg.batch_size}"
    assert T == cfg.seq_len, f"Seq mismatch: {T} vs {cfg.seq_len}"

    # -----------------------------
    # Extract activations
    # -----------------------------
    X, Y = extract_XY(
        model=model,
        input_ids=input_ids,
        layer_idx=layer_idx,
        cfg=cfg
    )

    # -----------------------------
    # Verify correctness
    # -----------------------------
    verify_dimensions(X, Y, cfg)

    # -----------------------------
    # Shuffle (important!)
    # -----------------------------
    if getattr(cfg, "shuffle_data", True):
        idx = np.random.permutation(X.shape[0])
        X = X[idx]
        Y = Y[idx]

    # -----------------------------
    # Optional train/val split
    # -----------------------------
    if hasattr(cfg, "train_split"):
        split = int(cfg.train_split * X.shape[0])

        return {
            "X_train": X[:split],
            "Y_train": Y[:split],
            "X_val": X[split:],
            "Y_val": Y[split:]
        }

    # Default
    return {
        "X": X,
        "Y": Y
    }


# ============================================================
# 6. UTILITY
# ============================================================

def _to_tensor_if_possible(input_ids, target_ids):
    """
    Convert to torch.Tensor if PyTorch is available
    """
    if torch is not None:
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_ids, dtype=torch.long),
        )
    else:
        return input_ids, target_ids