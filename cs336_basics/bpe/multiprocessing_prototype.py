from collections import Counter
import time
import regex as re

from cs336_basics.bpe.tokenizer_prototypes import find_chunk_boundaries

num_processes = 4  # Example number of processes

def pre_tokenize_and_count(
    task: tuple[bytes, dict[str, int], re.Pattern, re.Pattern]
) -> Counter:
    """
    Pre-tokenizes a chunk of text and counts the frequency of each word or special token.
    """
    bytes, special_token_vocabulary, special_tokens_regex, word_splitter_regex = task
    text = bytes.decode("utf-8", errors="ignore")
    special_tokens = set(special_token_vocabulary.keys())
    
    words_list = []
    
    # Split the text by the special tokens.
    if special_tokens_regex:
        pieces = special_tokens_regex.split(text)
    else:
        pieces = [text]

    for piece in pieces:
        if not piece:
            continue
            
        if piece in special_tokens:
            # If the piece is a special token, count its id.
            token_id = special_token_vocabulary[piece]
            words_list.append((token_id,))
        else:
            # The piece should be small, so no lazy loading is needed. find all words in it.
            for word_str in word_splitter_regex.findall(piece):
                if word_str:
                    byte_sequence = word_str.encode("utf-8")
                    id_sequence = tuple(byte_sequence)
                    words_list.append(id_sequence)

    return Counter(words_list)

# Dummy merge.
def merge_dummy(tokenization_table: Counter[int], steps: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  new_words = []
  merge_sequence = []  

  for _ in tqdm(range(steps), desc="Merge"):
    token_pair_freq_table = {}
    # Initialize the merge freq table by adding the byte pairs.
    for word in tokenization_table.keys():
      add_word_to_token_pair_freq_table(token_pair_freq_table, word, tokenization_table[word])
    
    # Find the largest entry by frequency, break ties by lexicographical value.
  
    # Naive implementation, using Python max, linear time.
    most_frequent_pair = max(token_pair_freq_table, key=lambda k, t=token_pair_freq_table: (t[k], k))
    merged_most_frequent_pair = most_frequent_pair[0] + most_frequent_pair[1]
    # print(f'most_frequent_pair is {most_frequent_pair}, frequency: {token_pair_freq_table[most_frequent_pair]}')
    merge_sequence.append((most_frequent_pair[0], most_frequent_pair[1]))
    # Record as a new word in the final vocabulary.
    new_words.append(merged_most_frequent_pair)

    # store words to be merged, and record the operations.
    operations = []
    for word in tokenization_table.keys():
      # Store the indices of the pair with max frequency, instead of doing the merge in-place.
      indices_appeareance = [i for i in range(len(word) - 1) if word[i:i + 2] == most_frequent_pair]
      # Commit all merges for this word.
      if indices_appeareance:
        new_word = []
        i = 0
        while i < len(word):
          if i in indices_appeareance:
            new_word.append(merged_most_frequent_pair)
            i += 2
          else:
            new_word.append(word[i])
            i += 1
        new_word = tuple(new_word)
        operations.append((word, new_word, tokenization_table[word]))
    
    for word, new_word, appearances in operations:
      # Avoid overwritting the the new word's appearances if it already exists.
      tokenization_table[new_word] = tokenization_table.get(new_word, 0) + appearances 
      del tokenization_table[word]

  return (dict(enumerate(STARTER_VOCABULARY + new_words)), merge_sequence)


# Initialization
start_time = time.time()
vocab = {i: bytes([i]) for i in range(256)}
PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
compiled_PAT = re.compile(PAT)

special_tokens = ["<|endoftext|>"]
# Add special tokens to vocabulary. 
# Also create a mapping from special token to its ID.
special_token_reference_vocabulary = {}
for token in special_tokens:
    token_bytes = token.encode("utf-8")
    if token_bytes not in vocab.values():
        vocab[len(vocab)] = token_bytes
        special_token_reference_vocabulary[token] = len(vocab) - 1

# Create a splitter of text, by the special tokens
special_tokens_sorted = sorted(
    [t.encode("utf-8") for t in special_tokens], key=len, reverse=True
)
escaped_tokens = [re.escape(t.decode("utf-8")) for t in special_tokens_sorted]
special_tokens_regex = "|".join(escaped_tokens)
if special_tokens_regex:
    special_token_pattern = re.compile(f"({special_tokens_regex})")

# Create a reference vocabulary for special tokens
special_token_to_id = {token: i for i, token in enumerate(special_tokens)}

tasks = []
with open("tests/fixtures/tinystories_sample.txt", "rb") as f:
    before_pretokenization_time = time.time()
    boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

    compiled_PAT = re.compile(PAT)
        
    # Create a list of tasks, for each chunk.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        tasks.append((chunk.encode("utf-8"), special_token_reference_vocabulary, special_token_pattern, compiled_PAT))

end_time = time.time()
print(f"Initialization time: {end_time - start_time:.2f} seconds")


def process_task(task) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    frequency_table = pre_tokenize_and_count(task)


    

merges = []

for task in tasks:
    updated_vocab, merge_sequence = process_task(task)
    merges