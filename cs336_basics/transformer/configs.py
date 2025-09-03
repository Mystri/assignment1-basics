from dataclasses import dataclass

import numpy as np



@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 512
    num_heads: int = 8
    rope_theta: float = 10000
    ffn_type: str = "GeLU"
    d_ff: int = 2048
    num_layers: int = 6
    context_length: int = 128
    

@dataclass
class DecodingConfig:
    max_new_tokens: int = 32
    temperature: float = 0.7
    # Top-p sampling.
    p: float = 0.9

    model: TransformerConfig | None = None


@dataclass
class TrainingConfig:
    # Scheduling
    steps: int = 10000
    num_words: int = 32
    batch_size: int = 64
    checkpoint_interval = 1000
    logging_interval = 1000

    # Tokenizer
    vocab_filepath: str = ""
    merges_filepath: str = ""
    special_tokens: list[str] | None = None

    # Model - Pure transformer (pre-norm + RoPE)
    model: TransformerConfig | None = None

    # Optimizer - AdamW + LR cosine decay
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay = 0.01
    eps: float = 1e-8
    warmup_iters: int = 4000
    cosine_cycle_iters: int = 50000

    # Gradient clipping
    max_l2_norm: float = 100

    # LR cosine scheduling
    max_learning_rate = 3e-4
    min_learning_rate = 3e-5
    warmup_iters = 4000
    cosine_cycle_iters = 50000

    # Checkpointing
    checkpoints_dir = "checkpoints"
