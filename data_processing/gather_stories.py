import json

with open('data/train_separated.jsonl', 'r') as infile, open('data/train_together.txt', 'w') as outfile:
    for line in infile:
        parsed_line = json.loads(line.strip())
        modified_line = parsed_line.strip() + "<|endoftext|>"
        outfile.write(modified_line)