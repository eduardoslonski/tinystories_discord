import pickle
from transformers import AutoTokenizer
import numpy as np

def check_separated(input_file, tokenizer):
    with open(input_file, 'rb') as infile:
        loaded_tokenized_data = pickle.load(infile)

    print(tokenizer.decode(loaded_tokenized_data[0]))

def check_together(input_file, tokenizer):
    loaded_data = np.fromfile(input_file, dtype=np.uint16)
    print(tokenizer.decode(loaded_data[:600]))

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('roneneldan/TinyStories-2Layers-33M')
    check_together('data/valid_together_tokenized.bin', tokenizer)