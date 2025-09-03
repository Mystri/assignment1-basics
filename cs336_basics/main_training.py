from cs336_basics.bpe.Tokenizer import Tokenizer
from cs336_basics.transformer.configs import TrainingConfig, TransformerConfig
from cs336_basics.transformer.training import train


if __name__ == "__main__":
    config = TrainingConfig()
    config.steps = 1000
    config.logging_interval = 10

    project_root = "/root/autodl-tmp/assignment1-basics/cs336_basics/"
    tokenizer = Tokenizer.from_files(
        vocab_filepath=f"{project_root}checkpoints/tokenizer/vocab_tinystories.pkl",
        merges_filepath=f"{project_root}checkpoints/tokenizer/merges_tinystories.pkl",
        special_tokens=["<|endoftext|>"],
    )
    config.model = TransformerConfig(vocab_size=len(tokenizer.vocab))

    # train small small batch
    with open('/root/autodl-tmp/assignment1-basics/cs336_basics/data/tinystories_sample.txt') as f:
        contents = f.read()

        train(data=tokenizer.encode(contents), config=config)
