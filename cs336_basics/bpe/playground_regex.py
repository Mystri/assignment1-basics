import regex as re

text = """Lucy was so happy and excited. She picked the prettiest flowers and took them home to show her mommy. When Mommy saw the flowers, she said they were very yummy. Lucy was so glad she could make her mommy happy! 

<|endoftext|>
Once upon a time,"""
print(f"text: '{text}'")

# Special token added to front
special_token = "<|endoftext|>"
special_token_escaped = re.escape(special_token)

# GPT-like BPE pattern
PAT = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

# Your pattern: special tokens in front
pattern_with_specials = f"(?:{special_token_escaped})|{PAT}"
pattern_without_specials = PAT

# Compile
re_with_specials = re.compile(pattern_with_specials)
re_without_specials = re.compile(pattern_without_specials)

# Run
print("🔹 Pattern WITH special tokens")
print([m.group() for m in re_with_specials.finditer(text)])

text_split = text.split(special_token)
print("\n🔸 Pattern WITHOUT special tokens")
for t in text_split:
    print([m.group() for m in re_without_specials.finditer()])
