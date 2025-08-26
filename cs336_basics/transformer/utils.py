from collections.abc import Iterable
import math
import torch
from torch import Tensor
from jaxtyping import Float


def cross_entropy(
    expected: Float[Tensor, " vocab_size"],
    logits: Float[Tensor, " ... seq_len vocab_size"],
    dim: int = -1,
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


def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):

    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it < cosine_cycle_iters:
        return min_learning_rate + 0.5 * (
            1
            + math.cos(
                (it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi
            )
        ) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate


def clip_gradient(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    # Approach - faster, that uses only one norm call.
    grads = [p.grad.flatten() for p in parameters if p.grad is not None]
    if len(grads) == 0:
        return

    flat_grads = torch.cat(grads)
    actual_l2_norm = flat_grads.norm(2)

    # Approach - more memory-efficient, that only adding scalars together.
    # for p in parameters:
    #     if p.grad is not None:
    #         param_norm = p.grad.norm(2)        # L2 norm of this tensor
    #         total_norm_sq += param_norm.item() ** 2
    # total_norm = total_norm_sq ** 0.5

    if actual_l2_norm > max_l2_norm:
        clip_coefficient = max_l2_norm / actual_l2_norm
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coefficient)
