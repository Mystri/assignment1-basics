import math
import torch
import einops


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einops.einsum(self.weight, x, "d_out d_in, ... d_in -> ... d_out")


class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        w = torch.empty(num_embeddings, embedding_dim)
        torch.nn.init.trunc_normal_(w, mean=0.0, std=1.0, a=-3, b=3)
        self.weight = torch.nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]


class RmsNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)  # Prevent overflow in mean-square calculation
        # Evaluate mean-square against the last dimension, for the input
        rms = torch.sqrt(torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + self.eps)
        # Einsum for element-wise multiplication
        result = (
            einops.einsum(self.weight, x, "d_input, ... d_input -> ... d_input") / rms
        )
        return result.to(in_dtype)


def silu(x: torch.Tensor):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        evens = x[..., 0::2]
        odds = x[..., 1::2]  # Split the input tensor into evens and odds

        result_evens = evens * cos - odds * sin
        result_odds = evens * sin + odds * cos

        # Interleave:
        # Move the first dimension (even, odd) to the last dimension.
        # The second to last dimension should contain a lot of (even, odd) pairs.
        pairs = einops.rearrange(
            torch.stack([result_evens, result_odds]), "d_even_odd ... -> ... d_even_odd"
        )
        result = einops.rearrange(pairs, "... d_2 d_1 -> ... (d_2 d_1)")

        return result
