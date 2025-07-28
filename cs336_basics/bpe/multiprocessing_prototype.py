from collections import Counter
import time
import regex as re

from cs336_basics.bpe.tokenizer_prototypes import find_chunk_boundaries

num_processes = 4  # Example number of processes

def pre_tokenize_and_count(
    task: tuple[bytes, dict[str, int], re.Pattern]
) -> Counter:
    bytes, special_token_vocabulary, token_splitter = task
    text = bytes.decode("utf-8", errors="ignore")
    special_tokens = set(special_token_vocabulary.keys())
    
    words_list = []
    
    if token_splitter:
        tokens = token_splitter.split(text)
    else:
        tokens = [text]

    for token in tokens:
        if not token:
            continue
            
        if token in special_tokens:
            token_id = special_token_vocabulary[token]
            words_list.append((token_id,))
        else:
            for word_str in token_splitter.findall(token):
                if word_str:
                    byte_sequence = word_str.encode("utf-8")
                    id_sequence = tuple(byte_sequence)
                    words_list.append(id_sequence)

    return Counter(words_list)

# Initialization
start_time = time.time()
vocab = {i: bytes([i]) for i in range(256)}
PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

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
special_token_regex = "|".join(escaped_tokens)
if special_token_regex:
    special_token_pattern = re.compile(f"({special_token_regex})")

# Create a reference vocabulary for special tokens
special_token_to_id = {token: i for i, token in enumerate(special_tokens)}

tasks = []
with open("tests/fixtures/tinystories_sample.txt", "rb") as f:
    before_pretokenization_time = time.time()
    boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

    compiled_PAT = re.compile(PAT)
        
    # Create a list of tasks for each chunk.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        tasks.append((chunk.encode("utf-8"), vocab, compiled_PAT))

end_time = time.time()
print(f"Initialization time: {end_time - start_time:.2f} seconds")