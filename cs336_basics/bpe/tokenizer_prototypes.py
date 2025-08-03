from typing import BinaryIO
import regex as re
from multiprocessing import Pool
import os
from collections import Counter, defaultdict
from tqdm import tqdm

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
EOT = "<|endoftext|>"
STARTER_VOCABULARY = [EOT.encode("utf-8")] + [bytes([i]) for i in range(256)]


def createTokenizer(
    input_path: str, vocab_size: int, special_tokens: list[str] = []
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    with open(input_path) as file:
        content = file.read()
        frequency_table = pretokenize_in_parallel(content)


"""
Get the boundary indices of the given file.
"""


def find_chunk_boundaries(
    file: BinaryIO, desired_num_chunks: int, split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            # Move to the end of mini chunk if
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_in_parallel(task: tuple[bytes, dict[str, int], re.Pattern]) -> Counter:
    chunk, special_token_to_id, delimiter_pattern = task


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_chunks: int = 8,
    num_processes: int = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes

    # read file
    with open(input_path, "rb") as f:

        # split to num_chunks chunks.
        boundaries = find_chunk_boundaries(f, num_chunks, special_tokens)

        # Organize them into tasks.
        tasks = []

    # map to processes
    all_word_freqs = Counter()
    start_time = time.time()
    with Pool(processes=num_processes) as pool:
        print(
            f"Starting pre-tokenization with {num_processes} processes on {len(tasks)} chunks..."
        )

        # Use imap_unordered, as we are processing . It returns an iterator.
        results_iterator = pool.imap_unordered(pretokenize_in_parallel, tasks)

        # Iterate through results as they complete and merge them one by one.
        for chunk_counter in tqdm(
            results_iterator, total=len(tasks), desc="Processing chunks"
        ):
            all_word_freqs.update(chunk_counter)

    end_time = time.time()
    print(
        f"Pre-tokenization and initial counting time: {end_time - start_time:.2f} seconds"
    )


# Dummy pretokenizer.
# Implements the pretokenizer, but takes a plain string and has no parellelism.
def pretokenize_dummy_tuple_bytes(
    input_str: str, special_tokens: list[str]
) -> dict[tuple[bytes], int]:
    frequency_table: dict[tuple[bytes]] = defaultdict(int)

    # Change the splitter Regex so that special tokens get directly feed into, instead of being split
    escaped_special_tokens = [re.escape(token) for token in special_tokens]
    special_token_patterns = "|".join(escaped_special_tokens)

    pattern_filtered_special_tokens = f"(?:{special_token_patterns})|{PAT}"

    for match in tqdm(
        re.finditer(pattern_filtered_special_tokens, input_str), desc="Pretokenization"
    ):
        match = match.group()

        if match in special_tokens:
            continue

        token = match.encode(encoding="utf-8")
        # Convert token to token of 'bytes', allow merging.
        token_byte_tuple = tuple(bytes([char]) for char in token)
        # Update the token's count
        frequency_table[token_byte_tuple] += 1
    return frequency_table


# Add all byte pairs of a word to a frequency table. For example, " lower" -> ' l', 'lo', 'ow', 'we', 'er'.
def add_word_to_token_pair_freq_table(merge_freq_table, word, word_freq):
    for i in range(len(word) - 1):
        pair = word[i : i + 2]  # Get the pair
        merge_freq_table[pair] = merge_freq_table.get(pair, 0) + word_freq


# Dummy merge.
def merge_dummy(
    tokenization_table: dict[tuple[bytes], int], steps: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    new_words = []
    merge_sequence = []

    for _ in tqdm(range(steps), desc="Merge"):
        token_pair_freq_table = {}
        # Initialize the merge freq table by adding the byte pairs.
        for word in tokenization_table.keys():
            add_word_to_token_pair_freq_table(
                token_pair_freq_table, word, tokenization_table[word]
            )

        # Find the largest entry by frequency, break ties by lexicographical value.

        # Naive implementation, using Python max, linear time.
        most_frequent_pair = max(
            token_pair_freq_table, key=lambda k, t=token_pair_freq_table: (t[k], k)
        )
        merged_most_frequent_pair = most_frequent_pair[0] + most_frequent_pair[1]
        # print(f'most_frequent_pair is {most_frequent_pair}, frequency: {token_pair_freq_table[most_frequent_pair]}')
        merge_sequence.append((most_frequent_pair[0], most_frequent_pair[1]))
        # Record as a new word in the final vocabulary.
        new_words.append(merged_most_frequent_pair)

        # store words to be merged, and record the operations.
        operations = []
        for word in tokenization_table.keys():
            # Store the indices of the pair with max frequency, instead of doing the merge in-place.
            indices_appeareance = [
                i for i in range(len(word) - 1) if word[i : i + 2] == most_frequent_pair
            ]
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
            tokenization_table[new_word] = (
                tokenization_table.get(new_word, 0) + appearances
            )
            del tokenization_table[word]

    return (dict(enumerate(STARTER_VOCABULARY + new_words)), merge_sequence)


# Sanity check for the dummy pretokenization/merge.
# with open(rf'{os.path.dirname(os.path.abspath(__file__))}\\simple_text.txt', 'r', encoding='utf-8') as file:
#   content = file.read()
#   pretokenization = pretokenize_dummy_tuple_bytes(content)
#   print(f'Dummy Pretokenization result: ${pretokenization}')
#   new_words, merge_sequence = merge_dummy(pretokenization, 6)
#   vocab = STARTER_VOCABULARY + new_words
#   print(f'Merge result - New words: {new_words}')
