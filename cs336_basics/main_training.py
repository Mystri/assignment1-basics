import numpy as np
from cs336_basics.bpe.Tokenizer import Tokenizer
from cs336_basics.transformer.configs import TrainingConfig, TransformerConfig
from cs336_basics.transformer.training import train


if __name__ == "__main__":
    config = TrainingConfig()

    project_root = "/root/autodl-tmp/assignment1-basics/cs336_basics/"
    tokenizer = Tokenizer.from_files(
        vocab_filepath=f"{project_root}checkpoints/tokenizer/vocab_tinystories.pkl",
        merges_filepath=f"{project_root}checkpoints/tokenizer/merges_tinystories.pkl",
        special_tokens=["<|endoftext|>"],
    )
    config.model = TransformerConfig(vocab_size=len(tokenizer.vocab))

    # train normal batch
    train_text = '/root/autodl-tmp/assignment1-basics/cs336_basics/data/TinyStoriesV2-GPT4-train.txt'
    valid_text = '/root/autodl-tmp/assignment1-basics/cs336_basics/data/TinyStoriesV2-GPT4-valid.txt'

    train_data = np.memmap(filename=train_text, mode="r")
    valid_data = np.memmap(filename=valid_text, mode="r")

    train(train_data=train_data, valid_data=valid_data, config=config, project_name="tinystories_sample")
