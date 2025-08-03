from cs336_basics.bpe.pretokenization_example import find_chunk_boundaries

with open("tests/fixtures/tinystories_sample.txt", "r") as f:
    a = find_chunk_boundaries(f, 2, b"<|endoftext|>")
    print(a)
