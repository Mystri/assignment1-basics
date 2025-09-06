from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 768
    num_heads: int = 12
    rope_theta: float = 10000
    ffn_type: str = "SwiGLU"
    d_ff: int = 1344
    num_layers: int = 12
    context_length: int = 256


def model_size(model: TransformerConfig) -> str:

    # Per transformer layer

    ffn_params_per_layer = 0
    if model.ffn_type == "SwiGLU":
        ffn_params_per_layer = 3 * model.d_model * model.d_ff

    attn_params_per_layer = 4 * model.d_model * model.d_model

    ln_params_per_layer = 2 * model.d_model

    # Final LN
    final_ln_params = model.d_model

    # Output/input embedding
    output_and_intput_embedding = 2 * model.d_model * model.vocab_size

    return f"Model size: {model.num_layers * (attn_params_per_layer + ffn_params_per_layer + ln_params_per_layer) + output_and_intput_embedding + final_ln_params:,}"


@dataclass
class DecodingConfig:
    max_new_tokens: int = 256
    temperature: float = 1
    # Top-p sampling.
    p: float = 0.9

    model: TransformerConfig | None = None


@dataclass
class TrainingConfig:
    # System
    checkpoint_interval = 1000
    logging_interval = 1000
    accumulation_steps = 4

    # Scheduling
    steps: int = 10000
    num_words: int = 32
    batch_size: int = 64

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


def peak_memory_cost(config: TrainingConfig, dtype: torch.dtype) -> str:
    model = config.model
    bsd = config.batch_size * model.context_length * model.d_model
    bhss = (
        config.batch_size
        * model.num_heads
        * model.context_length
        * model.context_length
    )
    bsn_word = config.batch_size * model.context_length * model.vocab_size
    bs = config.batch_size * model.context_length
    N = model.num_layers
    # SiLU: d_ff = 4d_model.

    # See Aug 25 notes.
    parameters = (
        2 * model.vocab_size * model.d_model
        + N * (12 * model.d_model * model.d_model + 2 * model.d_model)
        + model.d_model
    )
    optimizer_states = 2 * parameters
    gradients = 4 * parameters
    activations = bsd + N * (16 * bsd + 2 * bhss)

    bytes_per_param = 1
    if dtype == torch.float32:
        bytes_per_param = 4
    elif dtype == torch.float16:
        bytes_per_param = 2

    return f"Estimated peak memory cost with overhead 10%: {(parameters + optimizer_states + gradients + activations) * bytes_per_param * 1.1:,} bytes in fp32."  # FP32


if __name__ == "__main__":
    model = TransformerConfig(10000)
    print(model_size(model=model))
