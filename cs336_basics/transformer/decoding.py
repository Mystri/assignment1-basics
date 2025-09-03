from dataclasses import dataclass

import torch
from cs336_basics.bpe.Tokenizer import Tokenizer
from cs336_basics.transformer.configs import DecodingConfig
from cs336_basics.transformer.modules import Transformer
from cs336_basics.transformer.training import load_checkpoint


def softmax_with_temperature(x: torch.Tensor, temperature: float, dim: int = -1):
    x = torch.div(x, temperature)
    shifted_x = x - torch.max(x, dim=dim, keepdim=True).values

    exp_x_with_temp = torch.exp(shifted_x)
    return exp_x_with_temp / torch.sum(exp_x_with_temp, dim=dim, keepdim=True)


def decode_single_prompt(
    model: Transformer, tokenizer: Tokenizer, prompt: str, config: DecodingConfig
):
    eot = tokenizer.encode("<|endoftext|>")[0]
    tokenized_input = tokenizer.encode(prompt)
    context_length = model.context_length

    # Use nograd for torch inference.
    with torch.no_grad():
        for _ in range(config.max_new_tokens):

            # Get input tokens, whose length is no more than current context, from the prompt.
            context_length = model.context_length
            tokenized_input = (
                tokenized_input[-context_length:]
                if len(tokenized_input) >= context_length
                else tokenized_input
            )

            # Get tokenized output
            logits = model(tokenized_input)[-1, :]

            # Apply softmax with temperature
            words_prob_dist = softmax_with_temperature(
                logits, temperature=config.temperature
            )

            # Select based on top-p sampling
            sorted_prob_dist, corresponding_tokens = torch.sort(
                words_prob_dist, descending=True
            )
            # Find the cumulative sum (prefix sum) of the result distribution
            cumsum = torch.cumsum(sorted_prob_dist, dim=-1)
            # Find the index that has prefix sum of "p" of Top-p.
            cutoff_idx = torch.searchsorted(cumsum, config.p, )

            trimmed_prob_dist = sorted_prob_dist[:cutoff_idx]
            trimmed_tokens = corresponding_tokens[:cutoff_idx]

            # Normalize, since trimmed_prob_dist sums to p, not 1
            trimmed_prob_dist /= torch.sum(trimmed_prob_dist)

            # Sample the token from the sorted and get its token id (idx of the original output tensor).
            next_token_idx = torch.multinomial(trimmed_prob_dist, num_samples=1)
            # Convert to an actual token id, an int.
            next_token = trimmed_tokens[next_token_idx].item()

            if next_token == eot:
                break

            tokenized_input.append(next_token)
            print("Current output:")
            print(tokenizer.decode(tokenized_input))

    return tokenizer.decode(tokenized_input)
