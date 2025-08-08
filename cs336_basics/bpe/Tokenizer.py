from collections.abc import Iterable, Iterator
import pickle
import regex as re

from cs336_basics.bpe.train_bpe import Token, Word


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None,
    ):
        assert vocab
        # assert merges

        self.vocab = vocab

        self.token_id_of_word = {v: k for k, v in self.vocab.items()}

        # Easily determine a merge's priority in the whole sequence.
        self.merge_ranks = {
            # Convert bytes in merges to token IDs.
            (self.token_id_of_word[merge[0]], self.token_id_of_word[merge[1]]): i for i, merge in enumerate(merges)
        }  # also coule be used to determine if a pair is mergeable.

        PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        self.pretokenizer_pattern = re.compile(PAT)

        if special_tokens:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.special_pattern = (
                "(" + "|".join(re.escape(k) for k in self.special_tokens) + ")"
            )
        else:
            self.special_tokens = None
            self.special_pattern = None

    def encode(self, text: str) -> list[int]:
        """
        Encodes a string into a list of token IDs based on the vocabulary.
        """
        pre_tokenization_result = self.pretokenize(text)
        result = []
        for idx, word in enumerate(pre_tokenization_result):
            encode_word_result = self.encode_word(word)
            result.append(encode_word_result)
        
        
        return sum(result, []) # Flatten the list of [list of token IDs]

    def pretokenize(self, text: str) -> list[bytes]:
        """
        Pre-tokenizes the input text into a list of byte strings (tokens).
        Handles special tokens if provided.
        """
        if not self.special_tokens:
            return [
                self.convert_bytes_to_token_list(match.encode("utf-8")) for match in self.pretokenizer_pattern.findall(text)
            ]

        tokens = []
        # Split on special tokens first
        parts = re.split(self.special_pattern, text)
        for part in parts:
            if part in self.special_tokens:
                tokens.append([self.token_id_of_word[part.encode("utf-8")]])
            else:
                # Each find would be a word.
                matches = [
                    match.encode(encoding="utf-8")
                    for match in self.pretokenizer_pattern.findall(part)
                ]
                tokens += [
                    self.convert_bytes_to_token_list(match) for match in matches
                ]

        return tokens

    def convert_bytes_to_token_list(self, text: bytes) -> list[Token]:
        # Convert each byte of a word of a into the form of its token id.
        return [self.token_id_of_word[bytes([byte])] for byte in text]

    def encode_word(self, word: Word) -> list[int]:
        return word

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        Constructs a Tokenizer from a serialized vocab, list of merges, and (optionally) list of special tokens.
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Yields token IDs lazily from an iterable of strings (e.g., a file handle)."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        """
        Decodes a list of token IDs back into a string.
        This method assumes that the token IDs correspond to the vocabulary.
        """
        return b"".join(self.vocab[token_id] for token_id in token_ids).decode(
            "utf-8", errors="replace"
        )


if __name__ == "__main__":
    with open("data/tinystories_sample.txt") as f:
        text = f.read()
        tokenizer = Tokenizer.from_files(
            "cs336_basics/bpe/output/vocab_tinystories.pkl",
            "cs336_basics/bpe/output/merges_tinystories_sample.pkl",
            ["<|endoftext|>"]
        )

        pre_token_result = tokenizer.pretokenize(text)