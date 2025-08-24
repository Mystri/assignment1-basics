import torch
from torch import Tensor
from jaxtyping import Float


def cross_entropy(
    expected: Float[Tensor, " vocab_size"],
    logits: Float[Tensor, " ... seq_len vocab_size"],
    dim: int = -1
) -> Float:
    # Since it is the only event possible, because the training data is [0, 1, 0, 0] - onehot.
    # so only the -log (p_logits), of the expected word.
    max_logits = torch.max(logits, dim=dim, keepdim=True).values
    shifted_logits = logits - max_logits
    # top = log(exp(logits[expected]) - exp(logits.max)) = logits[expected] - logits.max
    # logits[expected]
    top = logits.gather(dim=dim, index=expected.unsqueeze(1))
    # logits[expected] - logits.max
    top -= max_logits

    # bottom = log(sum exp(logits) over every words)
    bottom = torch.log(torch.sum(torch.exp(shifted_logits), dim=dim))

    # Mean across all batches/sequences
    return -torch.mean(top - bottom)
