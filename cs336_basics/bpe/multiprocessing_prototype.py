from collections import Counter, defaultdict
import heapq
import time
import regex as re
from tqdm import tqdm

from cs336_basics.bpe.tokenizer_prototypes import find_chunk_boundaries

num_processes = 1  # Example number of processes

# Typedefs
Token = int  # Token is represented by an integer index in the vocabulary.
Word = tuple[Token]  # Represents that a word is a sequence of tokens.
Vocab = dict[int, bytes]


def pre_tokenize_and_count(
    task: tuple[bytes, dict[str, int], re.Pattern, str],
) -> Counter[Word]:
    """
    Pre-tokenizes a chunk of text and counts the frequency of each word or special token.
    """
    bytes, special_token_vocabulary, special_tokens_regex, word_splitter_regex = task
    text = bytes.decode("utf-8", errors="ignore")
    special_tokens = set(special_token_vocabulary.keys())

    words_list: list[Word] = []
    
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


class PQEntry:
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

    def get_merged_pair_bytes(self, vocab: dict[int, bytes]) -> tuple[bytes, bytes]:
        return (vocab[self.first], vocab[self.second])


class TokenPairPQ:
    def __init__(self, token_pairs: list[PQEntry] = []):
        heapq.heapify(token_pairs)
        self.pq = token_pairs

    def push(self, token_pair: PQEntry):
        heapq.heappush(self.pq, token_pair)
        self.pq.sort()  # Sort by frequency and lexicographical order

    def pop(self) -> PQEntry:
        return heapq.heappop(self.pq) if self.pq else None

    def is_empty(self) -> bool:
        return len(self.pq) == 0
    

class WordFreq: 
    def __init__(self, freq: int):
        self.freq = freq


class WordNode:
    def __init__(self, token: Token, word_freq: WordFreq):
        self.token = token
        self.word_freq = word_freq
        self.prev: WordNode = None
        self.next: WordNode = None


# Add all byte pairs of a word to a frequency table. For example, " lower" -> ' l', 'lo', 'ow', 'we', 'er'.
def initialize_word(
    word: Word,
    word_freq: int,
    token_pair_freq_table: dict[PQEntry, int],
    pair_to_words: dict[PQEntry, set[WordNode]],
):
    dummy_node = WordNode(None, 0)
    cursor = dummy_node
    word_freq = WordFreq(word_freq)

    for token in word:
        new_node = WordNode(token, word_freq)
        cursor.next = new_node
        new_node.prev = cursor
        cursor = new_node
    
    cursor = dummy_node.next
    
    # Traverse each pair in the word.
    while cursor and cursor.next:
        pair_node = cursor, cursor.next  # Get the pair, should be 2 Tokens.

        token_pair = pair_node[0].token, pair_node[1].token

        # Create a mapping from the pair to the words that contain it.
        pair_to_words[token_pair].add(cursor)

        # Update the frequency table.
        token_pair_freq_table[token_pair] += word_freq.freq

        # Move to the next pair.
        cursor = cursor.next


# Merge.
def merge(
    starter_vocab: Vocab, word_freq_table: Counter[Word], steps: int
) -> tuple[Vocab, list[PQEntry]]:
    new_words: list[PQEntry] = []
    merge_sequence: list[PQEntry] = []

    # Initialize the frequency table for token pairs.
    # Store tokenpair objects to quickly create the Priority Queue.
    token_pair_freq_table: dict[tuple[Token, Token], int] = defaultdict(int)

    # Create a map from each pair to their origins, so that each pair can refer back to the words that contained it.
    pair_occurrences_in_words: dict[tuple[Token, Token], set[WordNode]] = defaultdict(set)

    # Initialize the merge freq table by doing 2 things:
    # 1. count the pair by adding the word frequency to token_pair_freq_table.
    # 2. add the word to the words containing the pair, using the words_containing map.
    for word, freq in word_freq_table.items():
        initialize_word(word, freq, token_pair_freq_table, pair_occurrences_in_words)

    # Convert the frequency table to a priority queue of TokenPair objects.
    token_pair_pq = TokenPairPQ(
            [
                PQEntry(pair[0], pair[1], freq)
                for pair, freq in token_pair_freq_table.items()
            ]
        )

    result_vocab = starter_vocab.copy()

    for _ in tqdm(range(steps), desc="Merge"):

        # Pop the priority queue until we find a valid pair.
        most_frequent_pair = None
        while not token_pair_pq.is_empty():
            entry = token_pair_pq.pop()
            token_pair = (entry.first, entry.second)

            # If the pair is not in the frequency table, it means it has been merged or is invalid.
            if token_pair not in token_pair_freq_table:
                continue
            
            # The frequency table stores the latest frequency of the pair.
            # If the pair's frequncy matches the table, then it is the latest state of the queue and we should merge it.
            if (
                entry.frequency
                == token_pair_freq_table[token_pair]
            ):
                most_frequent_pair = entry
                break
        if most_frequent_pair is None:
            break

        # Record the merge operation for testing; nothing is really being merged now.
        merge_sequence.append(most_frequent_pair)

        # Record a new word in the final vocabulary.
        result_vocab[len(result_vocab)] = most_frequent_pair.get_merged_bytes(result_vocab)

        new_token = len(result_vocab) - 1  # The new token ID for the merged token.

        # store words to be merged, and record each operations of
        # "replacing the original word, and tranfer its frequency to the new word".
        for wordnode in pair_occurrences_in_words[
            most_frequent_pair
        ]:  # Update the words that contain the most frequent pair.
            # Store the indices of the pair with max frequency, instead of doing the merge in-place.
            
            old_first, old_second = wordnode, wordnode.next
            old_token_pair = (old_first.token, old_second.token)

            # Update the word.
            new_wordnode = WordNode(new_token, wordnode.word_freq)
            new_wordnode.prev = old_first.prev
            new_wordnode.next = old_second.next

            if old_first.prev:
                # 1. Modify the word.
                old_first.prev.next = new_wordnode
                new_wordnode.prev = old_first.prev

                # 2. Modify the map from pairs to words, since the word has changed.
                pair_occurrences_in_words[old_token_pair].discard(old_first)
                new_token_pair = PQEntry(
                    old_first.prev.token, new_token, wordnode.word_freq.freq
                )
                pair_occurrences_in_words[new_token_pair].add(old_first.prev)

                # 3. Modify the pair frequency table.
                # Reduce the frequency of the word from the old pair.
                token_pair_freq_table[old_token_pair] -= wordnode.word_freq.freq 
                # Increase the frequency of the word in the new pair.
                token_pair_freq_table[new_token_pair] += wordnode.word_freq.freq

                # Add the new token pair and its frequency to the priority queue.
                token_pair_pq.add(PQEntry(
                    old_first.prev.token, new_token, token_pair_freq_table[new_token_pair]
                ))

            if old_second.next:
                old_second.next.prev = new_wordnode
            
            del old_first
            del old_second

    return (result_vocab, merge_sequence)


# Initialization
start_time = time.time()
vocab: Vocab = {i: bytes([i]) for i in range(256)}
PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
compiled_PAT = re.compile(PAT)
file = "tests/fixtures/tinystories_sample.txt"

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
with open(file, "rb") as f:
    before_pretokenization_time = time.time()

    boundaries = find_chunk_boundaries(
        f, num_processes, "<|endoftext|>".encode("utf-8")
    )

    # Create a list of tasks, for each chunk.
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        tasks.append(
            (
                chunk.encode("utf-8"),
                special_token_reference_vocabulary,
                special_token_pattern,
                compiled_PAT,
            )
        )

print(f"Initialization time: {time.time() - start_time:.2f} seconds")

# start_time = time.time()
# # Simulates parallel pre-tokenization. todo: implement using multiprocessing.
# word_freq_table = Counter()
# for task in tasks:
#     word_freq_table.update(pre_tokenize_and_count(task))
# print(f"Pre-tokenization time: {time.time() - start_time:.2f} seconds")

# start_time = time.time()

# # Merge, which is not parallelizable.
# vocab, merges = merge(vocab, word_freq_table, steps=100)
# print(f"Merging time: {time.time() - start_time:.2f} seconds")
