from collections import Counter, defaultdict
import heapq
import time
import regex as re
from tqdm import tqdm

from cs336_basics.bpe.tokenizer_prototypes import find_chunk_boundaries

num_processes = 8  # Example number of processes

# Typedefs
Token = int # Token is represented by an integer index in the vocabulary.
Word = tuple[Token] # Represents that a word is a sequence of tokens.
Vocab = dict[int, bytes]

def pre_tokenize_and_count(
    task: tuple[bytes, dict[str, int], re.Pattern, re.Pattern]
) -> Counter[Word]:
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

# Add all byte pairs of a word to a frequency table. For example, " lower" -> ' l', 'lo', 'ow', 'we', 'er'.
def put_word_in_merge_freq_table(starter_vocab: Vocab, merge_freq_table: Counter[Token], word: str, word_freq: int, pair_to_words: defaultdict[set]):
  for i in range(len(word) - 1):
    pair_str = word[i:i + 2] # Get the pair, should be 2 characters.
    # Convert the pair to indices in the vocabulary.
    pair = TokenPair(starter_vocab[pair_str[0]], starter_vocab[pair_str[1]])

    # Create a mapping from the pair to the words that contain it.
    pair_to_words[pair].add(word)

    # Update the frequency table.
    merge_freq_table[pair] = merge_freq_table.get(pair, 0) + word_freq

class TokenPair:
    def __init__(self, first: Token, second: Token, frequency: int):
        self.first = first
        self.second = second
        self.frequency = frequency

    def __lt__(self, other):
        # Compare by frequency first, then lexicographically by the pairs.
        # Uses greaterthan for max-heap behavior.
        if self.frequency == other.frequency:
            return (self.first, self.second) > (other.first, other.second)
        return self.frequency > other.frequency

    def get_merged_bytes(self, vocab: dict[int, bytes]) -> bytes:
        return vocab[self.first] + vocab[self.second]
       
class TokenPairPQ:
    def __init__(self, token_pairs: list[TokenPair] = []):
        self.pq = heapq.heapify(token_pairs)

    def add(self, token_pair: TokenPair):
        heapq.heap_push(self.pq, token_pair)
        self.pq.sort()  # Sort by frequency and lexicographical order

    def pop(self) -> TokenPair:
        return heapq.heappop(self.pq) if self.pq else None

    def is_empty(self) -> bool:
        return len(self.pq) == 0

# Merge.
def merge(starter_vocab: Vocab, word_freq_table: Counter[Word], steps: int) -> tuple[dict[int, int], list[tuple[bytes, bytes]]]:
  new_words = []
  merge_sequence = []  

  token_pair_freq_table = defaultdict(int)

  # Create a map from each pair to their origins, so that each pair can refer back to the words that contained it.
  words_containing = defaultdict(set)

  # Initialize the merge freq table by initializing the byte pairs.
  for word in word_freq_table.values():
    put_word_in_merge_freq_table(starter_vocab, token_pair_freq_table, word, word_freq_table[word], words_containing)

  # Convert the frequency table to a priority queue of TokenPair objects.
  token_pair_pq = TokenPairPQ([TokenPair(first, second, freq) for (first, second), freq in token_pair_freq_table.items()])

  for _ in tqdm(range(steps), desc="Merge"):

    most_frequent_pair = token_pair_pq.pop()
    merged_most_frequent_pair = most_frequent_pair.get_merged_bytes(starter_vocab)

    # Record the merge operation for testing; nothing is really being merged now.
    merge_sequence.append((most_frequent_pair[0], most_frequent_pair[1]))

    # Record as a new word in the final vocabulary.
    new_words.append(merged_most_frequent_pair)

    # store words to be merged, and record each operations of 
    # "replacing the original word, and tranfer its frequency to the new word".
    operations = [] 
    for word in words_containing[TokenPair]: # Update the words that contain the most frequent pair.
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
        operations.append((word, new_word, word_freq_table[word]))
    
    # Apply the operations to the tokenization table.
    for word, new_word, appearances in operations:
      # Avoid overwritting the the new word's appearances if it already exists.
      word_freq_table[new_word] = word_freq_table.get(new_word, 0) + appearances 
      del word_freq_table[word]

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

print(f"Initialization time: {time.time() - start_time:.2f} seconds")

start_time = time.time()
# Simulates parallel pre-tokenization. todo: implement using multiprocessing.
for task in tasks:
    frequency_table = pre_tokenize_and_count(task)
print(f"Pre-tokenization time: {time.time() - start_time:.2f} seconds")

start_time = time.time()

# Merge, which is not parallelizable.
vocab, merges = merge(vocab, frequency_table, steps=1000)
print(f"Merging time: {time.time() - start_time:.2f} seconds")