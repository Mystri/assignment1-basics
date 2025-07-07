import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def createTokenizer(input_path:str, vocab_size:int, special_tokens:list[str]=[]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  frequency_table = pretokenize(input_path)

def pretokenize(input_path):
  pass

def pretokenize_dummy_tuple_bytes(input_path) -> dict[tuple[bytes], int]:
  re.finditer()

def merge_dummy():
  pass