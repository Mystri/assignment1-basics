# 2
## 2.1
1. What Unicode character does chr(0) return?
-  '\x00'
2. How does this character’s string representation (__repr__()) differ from its printed representation?
- `__repr__` shows its representation by its index in the encoding, '\x00'. 
- Directly printing this character renders it on screen which doesn't show anything.
3. What happens when this character occurs in text?
- When python directly renders the value of the char, it uses string repr
- When the char is passed to print it gets rendered using the printed repr
## 2.2
1. What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than
UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various
input strings.
- 