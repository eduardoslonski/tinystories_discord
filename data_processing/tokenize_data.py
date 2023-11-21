from transformers import AutoTokenizer
import json
import pickle
from tqdm import tqdm

def get_num_lines(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)

def tokenize_data(input_file, output_file):
    num_lines = get_num_lines(input_file)

    tokenizer = AutoTokenizer.from_pretrained('roneneldan/TinyStories-2Layers-33M')

    tokenized_data = []

    with open(input_file, 'r') as infile:
        for i, line in tqdm(enumerate(infile), total=num_lines):
            prefix = ''
            if i == 0:
                prefix = "<|endoftext|>"
            parsed_line = json.loads(line.strip())
            modified_line = prefix + parsed_line.strip() + "<|endoftext|>"
            text_tokenized = tokenizer.encode(modified_line)
            tokenized_data.append(text_tokenized)

    with open(output_file, 'wb') as outfile:
        pickle.dump(tokenized_data, outfile)

if __name__ == "__main__":
    tokenize_data('data/train_separated.jsonl', 'data/train_tokenized.pkl')