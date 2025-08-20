import math
from typing import Mapping
import torch
from torch import Tensor
import einops

from jaxtyping import Float, Bool, Int


class Linear(torch.nn.Module):
    """
    Basic linear fully-connected block, without a activation function.
    """

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

    def forward(self, x: Float[Tensor, "batch_size seq_len"]) -> Float[Tensor, "seq_len embedding_dim"]:
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
        # add epsilon to avoid sqrt on values too small and then division by 0, improving numerical stability.
        rms = torch.sqrt(torch.mean(torch.pow(x, 2), dim=-1, keepdim=True) + self.eps)
        # Intuition: x / meansquare(x)(norm) * gamma(weight) + beta(bias).
        # Same weight transformation being applied to each vector in the sequence/batch, so its affine.
        # We have gamma being learnable.
        # Einsum for element-wise multiplication
        result = (
            einops.einsum(self.weight, x, "d_input, ... d_input -> ... d_input") / rms
        )
        return result.to(in_dtype)


def silu(x: Tensor):
    # sigmoid: 1 / 1 + (e^-x)
    return x * torch.sigmoid(x)


class SWiGLU(torch.nn.Module):
    """
    SWiGLU-based linear unit.
    """

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

        # Split the input tensor into evens and odds
        evens: Float[Tensor, "... sequence_length d_k/2"] = x[..., 0::2]
        odds: Float[Tensor, "... sequence_length d_k/2"] = x[..., 1::2]  

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
    q: Float[Tensor, " ... q_seq_len   d_qk"],
    k: Float[Tensor, " ... kv_seq_len  d_qk"],
    v: Float[Tensor, " ... kv_seq_len  d_v"],
    mask: Bool[Tensor, " ... q_seq_len kv_seq_len"] | None = None,
) -> Float[Tensor, " ... d_v"]:
    """
    This term means exactly: the operation of attentionscore(x)v = softmax(qk/sqrt(d))v
    """
    # d_qk is the of dimension the input tensor.
    d_qk = q.shape[-1]
    # it is irrelevant to our attention score, which is "at token level", the attention between tokens, for example,
    # between 1st token and 5th token.
    # So, dimensions of the attention_score matrix should be q_seq_len kv_seq_len

    # Perform AttentionScore(q, k) = softmax(mask(qk) / sqrt(d))
    attention_scores = einops.einsum(
        q, k, "... q_seq_len  d_qk, ... kv_seq_len  d_qk -> ... q_seq_len kv_seq_len"
    ) / math.sqrt(d_qk)
    masked_attention_scores = torch.where(mask, attention_scores, float("-inf"))
    softmax_masked_attention_scores = softmax(masked_attention_scores, dim=-1)

    # Perform attention_score @ v.
    # Align the dimensions such that each query gets a value
    v = einops.einsum(
        softmax_masked_attention_scores,
        v,
        "... q_seq_len kv_seq_len,  ... kv_seq_len d_v -> ... q_seq_len d_v",
    )
    return v


class CausalMultiheadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        # As the original transformer described, d_model/h = d_qk = d_v
        self.d_head = d_model // num_heads

        self.Wqkv = Linear(
            out_features=3 * d_model, in_features=d_model, device=device, dtype=dtype
        )

        self.out_projection = Linear(
            out_features=self.d_model,
            in_features=self.d_model,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_v"],
        rope: RotaryPositionalEmbedding,
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, "... seq_len d_v"]:

        # self-attention, in_seq_len = out_seq_len
        seq_len = x.shape[-2]
        qkv = self.Wqkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Split into heads.
        q = einops.rearrange(
            q,
            "... seq_len (n_head d_head) -> ... n_head seq_len d_head",
            n_head=self.num_heads,
        )
        k = einops.rearrange(
            k,
            "... seq_len (n_head d_head) -> ... n_head seq_len d_head",
            n_head=self.num_heads,
        )
        v = einops.rearrange(
            v,
            "... seq_len (n_head d_head) -> ... n_head seq_len d_head",
            n_head=self.num_heads,
        )

        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = rope(q, token_positions)
            k = rope(k, token_positions)

        # Create Mask for the head view.
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )

        attention_result = scaled_dot_product_attention(q, k, v, mask)
        combined_attention_result = einops.rearrange(
            attention_result, "... n_head seq_len d_head -> ... seq_len (n_head d_head)"
        )
        return self.out_projection(combined_attention_result)
    

class PreNormTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # Write down the components of each transformer block.
        self.ln1 = RmsNorm(d_model=d_model, device=device, dtype=dtype)
        self.self_attention = CausalMultiheadSelfAttention(
            d_model=d_model, num_heads=num_heads, device=device, dtype=dtype
        )

        self.ln2 = RmsNorm(d_model=d_model, device=device, dtype=dtype)

        self.ffn = SWiGLU(d_ff=d_ff, d_model=d_model, device=device, dtype=dtype)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_model"],
        rope: RotaryPositionalEmbedding = None,
        token_positions: Float[Tensor, "seq_len"] | None = None,
    ):
        x = x + self.self_attention.forward(self.ln1(x), rope=rope, token_positions=token_positions)
        x = x + self.ffn.forward(self.ln2(x))
        return x

class Transformer(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length,
        rope_theta,
        num_layers,
        d_model,
        num_heads,
        d_ff,
    ):
        super().__init__()

        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
    
        self.rope = RotaryPositionalEmbedding(theta=rope_theta, d_k=d_model//num_heads, max_seq_len=context_length,)

        self.layers = torch.nn.ModuleList([
            PreNormTransformerBlock(d_model=d_model,num_heads=num_heads,d_ff=d_ff,)
            for _ in range(num_layers)
        ])

        self.ln_final = RmsNorm(d_model=d_model,)
        self.lm_head = Linear(out_features=vocab_size, in_features=d_model)


    def forward(
        self,
        in_indices: Int[Tensor, "... seq_len"],
        token_positions: Int[Tensor, "... seq_len"] | None = None,
    ) -> Float[Tensor, "seq_len d_model"]:
        embedding = self.token_embeddings(in_indices)

        for layer in self.layers:
            embedding = layer(embedding, self.rope, token_positions)
        
        embedding = self.ln_final(embedding)
        output = self.lm_head(embedding)
        return output

def remap_transformer_weights(weights: Mapping[str, any], num_layers):
    for idx in range(num_layers):
        q = weights.pop(f'layers.{idx}.attn.q_proj.weight')
        k = weights.pop(f'layers.{idx}.attn.k_proj.weight')
        v = weights.pop(f'layers.{idx}.attn.v_proj.weight')
        Wqkv = torch.cat((q, k, v), dim=0)
        weights[f'layers.{idx}.self_attention.Wqkv.weight'] = Wqkv

        weights[f'layers.{idx}.self_attention.out_projection.weight'] = weights.pop(f'layers.{idx}.attn.output_proj.weight')
        
    
    return weights