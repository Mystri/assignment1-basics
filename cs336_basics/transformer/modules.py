import math
import torch
from torch import Tensor
import einops

from jaxtyping import Float, Bool


def create_and_init_with_trunc_normal(mean, std, out_features, in_features):
    w = torch.empty(out_features, in_features)
    torch.nn.init.trunc_normal_(w, mean=mean, std=std, a=-3 * std, b=3 * std)
    return w


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()

        mean = 0.0
        std = math.sqrt(2 / (in_features + out_features))

        w = torch.empty(out_features, in_features)
        torch.nn.init.trunc_normal_(w, mean=mean, std=std, a=-3 * std, b=3 * std)

        self.weight = torch.nn.Parameter(w)

    def forward(self, x: Tensor) -> Tensor:
        return einops.einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        w = torch.empty(num_embeddings, embedding_dim)
        torch.nn.init.trunc_normal_(w, mean=0.0, std=1.0, a=-3, b=3)
        self.weight = torch.nn.Parameter(w)

    def forward(self, x: Float[Tensor, "vocab_size d_model"]) -> Tensor:
        # select [vocab_size * d_model](last dim, is vectors of indices) values as tensor from the embedding table.
        return self.weight[x]


class RmsNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # Prevent overflow in mean-square calculation
        # Evaluate mean-square against the last dimension, for the input
        rms = torch.sqrt(torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + self.eps)
        # Einsum for element-wise multiplication
        result = (
            einops.einsum(self.weight, x, "d_input, ... d_input -> ... d_input") / rms
        )
        return result.to(in_dtype)


def silu(x: Tensor):
    # sigmoid: 1 / 1 + (e^-x)
    return x * torch.sigmoid(x)


class SWiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super().__init__()

        if d_ff == None:
            d_ff = d_model * 8 // 3
            d_ff = math.ceil(d_ff / 64) * 64  # Ensure d_ff is a multiple of 64

        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        # Create a vector, each encodes the position of the possible input.
        positions = torch.arange(max_seq_len, device=device).unsqueeze(1)

        # For pos i and vector pair k, rotate of i * (1 / theta**(2k/d)).
        # 2 * k / d_k
        freqs_for_each_pair = torch.arange(0, d_k, 2, device=device) / d_k
        inv_freqs = theta**-freqs_for_each_pair
        angles = positions * inv_freqs

        self.register_buffer("cos", angles.cos().to(dtype), persistent=False)
        self.register_buffer("sin", angles.sin().to(dtype), persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... sequence_length d_k"],
        token_positions: Float[Tensor, "... sequence_length"],
    ) -> Float[Tensor, "... sequence_length d_k"]:
        cos: Float[Tensor, "... sequence_length"] = self.cos[token_positions]
        sin: Float[Tensor, "... sequence_length"] = self.sin[token_positions]

        evens: Float[Tensor, "... sequence_length d_k/2"] = x[..., 0::2]
        odds: Float[Tensor, "... sequence_length d_k/2"] = x[
            ..., 1::2
        ]  # Split the input tensor into evens and odds

        result_evens: Float[Tensor, "... sequence_length d_k/2"] = (
            evens * cos - odds * sin
        )
        result_odds: Float[Tensor, "... sequence_length d_k/2"] = (
            evens * sin + odds * cos
        )

        # Interleave:
        # Move the first dimension (even, odd) to the last dimension.
        # The second to last dimension should contain a lot of (even, odd) pairs.
        pairs: Float[Tensor, "... sequence_length d_k/2 2"] = einops.rearrange(
            torch.stack([result_evens, result_odds]), "d_even_odd ... -> ... d_even_odd"
        )
        result: Float[Tensor, "... sequence_length d_k"] = einops.rearrange(
            pairs, "... d_2 d_1 -> ... (d_2 d_1)"
        )

        return result


def softmax(x: Float[Tensor, " ... dim"], dim: int) -> Float[Tensor, " ... dim"]:
    shifted_x = x - torch.max(x, dim=dim, keepdim=True).values
    exp_x = torch.exp(shifted_x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: Float[Tensor, "batch_size ... q_seq_len   d_qk"],
    k: Float[Tensor, "batch_size ... kv_seq_len  d_qk"],
    v: Float[Tensor, "batch_size ... kv_seq_len  d_v"],
    mask: Bool[Tensor, "batch_size ... d_v"] | None = None,
) -> Float[Tensor, "batch_size ... d_v"]:
    """
    This term means exactly: the operation of attentionscore(x)v = softmax(qk/sqrt(d))v
    """
    d_qk = q.shape[-1]

    attention_score = softmax(
        einops.einsum(
            q,
            k,
            "... q_seq_len  d_qk, ... kv_seq_len  d_qk -> ... q_seq_len kv_seq_len",
        )
        / math.sqrt(d_qk)
    )[mask]

    v = einops.einsum(attention_score, v, "... q_seq_len kv_seq_len,  ... kv_seq_len, d_v -> ... d_v")
    return v
