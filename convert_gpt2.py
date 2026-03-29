import struct
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

with open('gpt2_tokenizer.bin', 'rb') as f:
    n_vocab = struct.unpack('I', f.read(4))[0]
    print(f"Vocab size: {n_vocab}\n")
    
    print("First 20 tokens:")
    for i in range(20):
        len_bytes = f.read(4)
        length = struct.unpack('I', len_bytes)[0]
        tok_bytes = f.read(length)
        tok_str = tok_bytes.decode('utf-8', errors='replace')
        print(f"  {i:5d}: {repr(tok_str)}")