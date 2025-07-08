import regex as re
import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOT = '<|endoftext|>'
STARTER_VOCABULARY = [EOT] + [bytes([i]) for i in range(256)]

def createTokenizer(input_path:str, vocab_size:int, special_tokens:list[str]=[]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  with open(input_path) as file:
    content = file.read()
    frequency_table = pretokenize(content)

def pretokenize(input_path):
  pass

# Dummy pretokenizer.
# Implements the pretokenizer, but takes a plain string and has no parellelism.
def pretokenize_dummy_tuple_bytes(input_str: str) -> dict[tuple[bytes], int]:
  frequency_table: dict[tuple[bytes]] = {}
  for match in re.finditer(PAT, input_str):
    token = match.group().encode(encoding='utf-8')
    # Convert token to token of 'bytes', allow merging.
    token_byte_tuple = tuple([bytes([char]) for char in token])
    # Update the token's count
    frequency_table[token_byte_tuple] = frequency_table.get(token_byte_tuple, 0) + 1
  return frequency_table

# Add all byte pairs of a word to a frequency table. For example, " lower" -> ' l', 'lo', 'ow', 'we', 'er'.
def add_word_to_token_pair_freq_table(merge_freq_table, word):
  for i in range(len(word) - 1):
    pair = word[i:i + 2] # Get the pair
    merge_freq_table[pair] = merge_freq_table.get(pair, 0) + 1

# Dummy merge.
def merge_dummy_naive(frequency_table: dict[tuple[bytes], int], steps: int):
  token_pair_freq_table = {}

  # Initialize the merge freq table by adding the byte pairs.
  for word in frequency_table.keys():
    add_word_to_token_pair_freq_table(token_pair_freq_table, word)

  for _ in range(steps):
    # Find the largest entry by frequency, break ties by lexicographical value.

    # Naive implementation, using Python max.
    most_frequent_pair = max(token_pair_freq_table, key=lambda k: (token_pair_freq_table[k], k))
    for word in frequency_table.keys():
      for i in range(len(word) - 1):
        if word[i] == most_frequent_pair[0] and word[i + 1] ==  most_frequent_pair[1]:
          print(f'Merging word {word[i:i + 2]}')
          merged_token = word[i] + word[i + 1]
          merged_word = word[:i] + tuple(merged_token) + word[i:]
          frequency_table[merged_word] = frequency_table[word]
          frequency_table[word] = -1
          
# Sanity check for the dummy pretokenization/merge.
with open(rf'{os.path.dirname(os.path.abspath(__file__))}\\simple_text.txt', 'r', encoding='utf-8') as file:
  content = file.read()
  pretokenization = pretokenize_dummy_tuple_bytes(content)
  print(f'Dummy Pretokenization result: ${pretokenization}')
  merge_dummy_naive(pretokenization, 1)

def merge_dummy():
  pass