from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Reproducibility
    seed: int = 42
 
    # Model (mamba-minimal)
    d_model: int = 256
    d_state: int = 16
    n_layers: int = 4
    vocab_size: int = 50257  # GPT-2 tokenizer
    expand: int = 2          # d_inner = d_model * expand = 512

    # Data
    dataset_name: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    seq_len: int = 128        # T — sequence length (tokens per sample)
    batch_size: int = 16      # B
    n_tokens_total: int = 10000  # contiguous token subset for layer experiments
    target_layer: int = 1     # which MAMBA layer to extract activations from (0-indexed)

    # Solver
    adam_lr_values: list = field(default_factory=lambda: [1e-2, 1e-3, 1e-4])
    adam_n_steps: int = 500
    adam_best_lr: float = 1e-3  # updated after lr sweep

    # Anchor sweep
    # k_frac = 1 means k = n (=d_inner=512); others are fractions
    anchor_k_fractions: list = field(default_factory=lambda: [1, 0.25, 0.5, 0.75])
    anchor_strategies: list = field(default_factory=lambda: ["random", "low_uncertainty", "high_uncertainty"])

    # Hybrid training
    hybrid_k_steps: list = field(default_factory=lambda: [5, 10, 20, 50])
    hybrid_train_steps: int = 1000
    hybrid_eval_every: int = 50
    hybrid_lr: float = 1e-3  # Adam lr for hybrid training

    # Paths
    output_dir: Path = Path("outputs")
    cache_dir: Path = Path(".cache")

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def d_inner(self) -> int:
        return self.d_model * self.expand
