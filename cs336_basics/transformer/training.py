from torch import Tensor
import numpy as np

def load_data(x, batch_size: int, context_length: int) -> tuple[Tensor, Tensor]:
    """
    Takes a numpy array of token IDs, batches them, and returns a pair of tensors:
    - (batch_size, context_length) — the actual batch
    - (batch_size, context_length) — the next token ID for each sample in the batch
    """
    
    

    x_sequences = np.zeros((batch_size, context_length), dtype=np.int64)
    y_sequences = np.zeros((batch_size, context_length), dtype=np.int64)