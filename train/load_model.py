import torch
from model import GPTModel
from config import GPTConfig
from transformers import AutoTokenizer
from generate import generate
import time

def main():
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    config = GPTConfig(vocab_size=tokenizer.eos_token_id+1)

    checkpoint = torch.load("checkpoints/checkpoint_16524.pt", map_location=torch.device(config.device))

    model = GPTModel(config).to(config.device)
    model.load_state_dict(checkpoint['model_state_dict'])

    input_text = "Once upon a time Sara and Ben went to the park"
    input_tokenized = tokenizer.encode(input_text, return_tensors="pt").to(config.device)
    start_time = time.time()
    generated = generate(model, input_tokenized, max_length=20, method="greedy", top_k=None, top_p=None, temperature=1.0).squeeze(0)
    print("time to generate", time.time() - start_time)
    print(tokenizer.decode(generated))

if __name__ == "__main__":
    main()