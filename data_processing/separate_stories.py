from tqdm import tqdm
import json

def get_num_lines(file_path):
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)

def split_stories(input_file, output_file, delimiter="<|endoftext|>"):
    buffer = ""
    num_lines = get_num_lines(input_file)

    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in tqdm(infile, total=num_lines):
            buffer += line
            if delimiter in buffer:
                story, buffer = buffer.split(delimiter, 1)
                outfile.write(f"{json.dumps(story.strip())}\n")

if __name__ == "__main__":
    split_stories("data/train.txt", "data/train_separated.jsonl")