from dataclasses import dataclass
import os
from torch import Tensor
import torch
import numpy as np
from datetime import datetime

from cs336_basics.bpe.tokenizer import Tokenizer
from cs336_basics.transformer.modules import Transformer
from cs336_basics.transformer.optimizers import AdamW
from cs336_basics.transformer.utils import (
    clip_gradient,
    cross_entropy,
    lr_cosine_schedule,
)


def get_batch(
    x: np.ndarray, batch_size: int, context_length: int
) -> tuple[Tensor, Tensor]:
    """
    Takes a numpy array of token IDs, batches them, and returns a pair of tensors:
    - (batch_size, context_length) — the actual batch
    - (batch_size, context_length) — the next token ID for each sample in the batch

    - This function works with both regular in-memory numpy arrays and
      memory-mapped arrays. load them with (np.memmap or np.load(..., mmap_mode="r")).
    - Usage example:
        # If dataset fits in memory
        x = np.load("tokens.npy")

        # If dataset is very large (lazy loading from disk)
        x = np.load("tokens.npy", mmap_mode="r")
        # or
        x = np.memmap("tokens.dat", dtype=np.int64, mode="r", shape=(total_tokens,))

        input_batch, next_batch = load_data(x, batch_size=32, context_length=128)
    - Ensure that the dtype of the loaded array matches the saved,
      and verify that token values are within the expected vocabulary size.
    """

    # We need to generate a whole context_length of training batch, so have to slice x starting from
    # somewhere before max_start_idx
    max_start_idx = len(x) - context_length - 1

    assert max_start_idx > 0

    # Sample batch_size times.
    starting_indices = np.random.randint(0, max_start_idx + 1, size=batch_size)

    input_sequences_np = np.zeros((batch_size, context_length), dtype=np.int64)
    next_tokens_np = np.zeros((batch_size, context_length), dtype=np.int64)

    for index_batch, start in enumerate(starting_indices):
        input_sequences_np[index_batch] = x[start : start + context_length]
        next_tokens_np[index_batch] = x[start + 1 : start + context_length + 1]

    input_sequences_tensor = torch.from_numpy(input_sequences_np)
    next_tokens_tensor = torch.from_numpy(next_tokens_np)

    return input_sequences_tensor, next_tokens_tensor


def save_checkpoint(model, optimizer, iteration, out) -> None:
    orig_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    obj = {
        "model": orig_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)


def load_checkpoint(src, model, optimizer):

    obj = torch.load(src)

    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])

    return obj["iteration"]


@dataclass
class TrainingConfig:
    # Scheduling
    training_data: np.ndarray
    num_words: int
    batch_size: int = 64
    context_length: int = 128
    checkpoint_interval = (1000,)

    # Tokenizer
    vocab_filepath: str = ""
    merges_filepath: str = ""
    special_tokens: list[str] | None = (None,)

    # Model - Pure transformer (pre-norm + RoPE)
    vocab_size: int
    d_model: int = 512
    num_heads: int = 8
    rope_theta: float = 10000
    ffn_type: str = "gelu"
    d_ff: int = 2048
    num_layers: int = 6

    # Optimizer - AdamW + LR cosine decay
    learning_rate: float = 3e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay = 0.01
    eps: float = 1e-8
    warmup_iters: int = 4000
    cosine_cycle_iters: int = 50000

    # Gradient clipping
    max_l2_norm: float

    # LR cosine scheduling
    max_learning_rate = 3e-4
    min_learning_rate = 3e-5
    warmup_iters = 4000
    cosine_cycle_iters = 50000

    # Checkpointing
    checkpoints_dir = "checkpoints"


def train(
    config: TrainingConfig,
    epochs: int,
):

    model = Transformer(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        rope_theta=config.rope_theta,
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
        eps=config.eps,
    )

    train_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_folder = f"{config.checkpoint_dir}/train_{train_time}"
    os.makedirs(train_folder, exist_ok=True)

    for epoch in range(epochs):
        # Create training batches
        batches = get_batch(config.training_data, config.batch_size, config.context_length)
        for step, batch in enumerate(batches):

            input_sequences_tensor, next_tokens_tensor = batch

            # Reset optimizer grads - each grad produced by a batch are independent.
            optimizer.zero_grad()
            # Run Forward.
            output = model(input_sequences_tensor)
            # Get Loss.
            loss = cross_entropy(expected=next_tokens_tensor, logits=output)
            # Run Backward. Pytorch will automatically set the corresponding nodes in the compute graph.
            loss.backward()
            # Clip model.
            clip_gradient(model.parameters(), max_l2_norm=config.max_l2_norm)

            # step optimizer.
            optimizer.step()
            # step lr scheduler.
            new_lr = lr_cosine_schedule(
                step,
                max_learning_rate=config.max_learning_rate,
                min_learning_rate=config.min_learning_rate,
                warmup_iters=config.warmup_iters,
                cosine_cycle_iters=config.cosine_cycle_iters,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr
            # Store
            if step % config.checkpoint_interval == 0:
                # Store a checkpoint.
                checkpoint_filename = f"epoch_{epoch}_step_{step}.pt"
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    iteration=step,
                    out=os.path.join(config.train_folder, checkpoint_filename),
                )


def decode():
    pass
