from dataclasses import dataclass
import os
from torch import Tensor
import torch
import numpy as np
from datetime import datetime

from tqdm import tqdm

from cs336_basics.transformer.configs import (
    TrainingConfig,
    model_size,
    peak_memory_cost,
)
from cs336_basics.transformer.modules import Transformer
from cs336_basics.transformer.optimizers import AdamW
from cs336_basics.transformer.utils import (
    clip_gradient,
    cross_entropy,
    lr_cosine_schedule,
)

import swanlab


def get_batch(
    x: np.ndarray, batch_size: int, context_length: int, device=None, dtype=None
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

        input_batch, next_batch = get_batch(x, batch_size=32, context_length=128)
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

    input_sequences_tensor = torch.from_numpy(input_sequences_np).to(
        device=device, dtype=dtype
    )
    next_tokens_tensor = torch.from_numpy(next_tokens_np).to(device=device, dtype=dtype)

    return input_sequences_tensor, next_tokens_tensor


def save_checkpoint(model, optimizer, iteration, out) -> None:
    orig_model = model._orig_mod if hasattr(model, "_orig_mod") else model

    obj = {
        "model": orig_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)


def load_checkpoint(src, model, optimizer=None):

    obj = torch.load(src)

    model.load_state_dict(obj["model"])
    if optimizer:
        optimizer.load_state_dict(obj["optimizer"])

    return obj["iteration"]


def train(
    train_data: np.memmap,
    valid_data: np.memmap,
    config: TrainingConfig,
    project_name="",
):

    # Swanlab init
    swanlab.init(project=project_name, config=config.__dict__)  # Log your configuration

    # Set up device and print drivers info.
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        max_cached = torch.cuda.max_memory_reserved() / 1024**3
        total_memory = torch.cuda.get_device_properties().total_memory / 1024**3
        free_memory = total_memory - allocated
        
        print(f"PyTorch Allocated: {allocated:.2f} GB")
        print(f"PyTorch Cached: {cached:.2f} GB")
        print(f"PyTorch Max Allocated: {max_allocated:.2f} GB")
        print(f"PyTorch Max Cached: {max_cached:.2f} GB")
        print(f"Free: {free_memory:.2f} GB")
        print(f"Total: {total_memory:.2f} GB")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA capability: {torch.cuda.get_device_capability()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Initialize the model and optimizer.
    model = Transformer(config.model)
    model.to(device=device, dtype=torch.half)
    print(f"Model size: {model_size(config.model)}")
    print(
        f"Estimated training peak memory cost: {peak_memory_cost(config, dtype=torch.half)}"
    )

    optimizer = AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
        eps=config.eps,
    )

    train_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_working_directory = f"{config.checkpoints_dir}/train_{train_time}"
    os.makedirs(train_working_directory, exist_ok=True)

    print("Training starts:")

    # Create training batches
    for step in tqdm(range(config.steps), desc=f"Training steps"):
        batches = get_batch(
            train_data,
            config.batch_size,
            config.model.context_length,
            device=device,
            dtype=None,
        )

        input_sequences_tensor, next_tokens_tensor = batches

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
            checkpoint_filename = f"step_{step}.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step,
                out=os.path.join(train_working_directory, checkpoint_filename),
            )
            # Log checkpoint as artifact
            # artifact = wandb.Artifact(
            #     name=f"model-checkpoint-{step}",
            #     type="model",
            #     description=f"Model checkpoint at step {step}"
            # )
            # artifact.add_file(checkpoint_path)
            # wandb.log_artifact(artifact)

        # Print progress.
        if step % config.logging_interval == 0:
            print(
                f"Step {step} | "
                f"Global Step {step} | "
                f"Loss: {loss:.4f} | "
                f"LR: {new_lr:.6f}"
            )
            
        checkpoint_filename = "final.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            iteration=step,
            out=os.path.join(train_working_directory, checkpoint_filename),
        )
    swanlab.finish()
