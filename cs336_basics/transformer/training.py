from torch import Tensor
import torch
import numpy as np


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
