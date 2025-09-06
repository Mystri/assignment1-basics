from cs336_basics.bpe.Tokenizer import Tokenizer

from cs336_basics.transformer.configs import DecodingConfig, TransformerConfig
from cs336_basics.transformer.decoding import decode_single_prompt
from cs336_basics.transformer.modules import Transformer
from cs336_basics.transformer.training import load_checkpoint


if __name__ == "__main__":
    project_root = "/root/autodl-tmp/assignment1-basics/cs336_basics/"

    vocab_filepath=f"{project_root}checkpoints/tokenizer/vocab_tinystories.pkl"
    merges_filepath=f"{project_root}checkpoints/tokenizer/merges_tinystories.pkl"

    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_filepath,
        merges_filepath=merges_filepath,
        special_tokens=["<|endoftext|>"],
    )

    print(tokenizer.encode("<|endoftext|>"))

    config = DecodingConfig()

    model = Transformer(config=TransformerConfig(vocab_size=len(tokenizer.vocab)))

    checkpoint = f"{project_root}checkpoints/train_20250906_214727/step_3000.pt"
    load_checkpoint(checkpoint, model)
    while True:
        decode_single_prompt(model=model, tokenizer=tokenizer, prompt="The ", config=config)
