import regex as re
import os

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOT = b'<|endoftext|>'
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
def add_word_to_token_pair_freq_table(merge_freq_table, word, word_freq):
  for i in range(len(word) - 1):
    pair = word[i:i + 2] # Get the pair
    merge_freq_table[pair] = merge_freq_table.get(pair, 0) + word_freq

# Dummy merge.
def merge_dummy(tokenization_table: dict[tuple[bytes], int], steps: int) -> list[bytes]:
  new_words = set()
  merge_sequence = []
  for _ in range(steps):
    token_pair_freq_table = {}
    # Initialize the merge freq table by adding the byte pairs.
    for word in tokenization_table.keys():
      add_word_to_token_pair_freq_table(token_pair_freq_table, word, tokenization_table[word])
    
    # Find the largest entry by frequency, break ties by lexicographical value.
  
    # Naive implementation, using Python max.
    most_frequent_pair = max(token_pair_freq_table, key=lambda k: (token_pair_freq_table[k], k))
    merged_most_frequent_pair = most_frequent_pair[0] + most_frequent_pair[1]
    # store words. to be optimized by indices
    words_need_merge = []
    for word in tokenization_table.keys():
      # Store the indices of the pair with max frequency, instead of doing the merge in-place.
      for i in range(len(word) - 1):
        if word[i] == most_frequent_pair[0] and word[i + 1] ==  most_frequent_pair[1]:
          print(f'Recording token merge for {word[i:i + 2]}, at position {i}')
          words_need_merge.append(word)
    
    # Merge the words.
    for word in words_need_merge:
      for i in range(len(word) - 1):
        if word[i] == most_frequent_pair[0] and word[i + 1] ==  most_frequent_pair[1]:
          print(f'Recording token merge for {word[i:i + 2]}, for word {word}')
          merged_word = (*word[:i], merged_most_frequent_pair, *word[i + 2:])
          new_words.add(merged_most_frequent_pair)
          tokenization_table[merged_word] = tokenization_table.pop(word)
    print(f"Table after {steps} merge: {tokenization_table}")

  return (list(new_words), merge_sequence)


# Sanity check for the dummy pretokenization/merge.
with open(rf'{os.path.dirname(os.path.abspath(__file__))}\\simple_text.txt', 'r', encoding='utf-8') as file:
  content = file.read()
  pretokenization = pretokenize_dummy_tuple_bytes(content)
  print(f'Dummy Pretokenization result: ${pretokenization}')
  new_words = merge_dummy(pretokenization, 6)
  vocab = STARTER_VOCABULARY + new_words
  print(f'Merge result - New words: {new_words}')

def merge_dummy():
  pass