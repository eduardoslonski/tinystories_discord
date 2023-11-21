import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from model import GPTModel
from config import GPTConfig
from dataset import GPTDataset
from trainer import Trainer
from utils_training import format_magnitude

def get_train_obj(config):
    model = GPTModel(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    return model, optimizer, loss_fn

def get_dataloaders(config):
    train_perc = 1
    val_perc = 0.05
    
    dataset_train = GPTDataset("data/train.bin", config.context_length)
    dataset_train = Subset(dataset_train, range(int(train_perc * len(dataset_train))))
    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=False)

    dataset_val = GPTDataset("data/valid.bin", config.context_length)
    dataset_val = Subset(dataset_val, range(int(val_perc * len(dataset_val))))
    dataloader_val = DataLoader(dataset_val, batch_size=config.batch_size_val, shuffle=False)

    train_size = len(dataloader_train)
    val_size = len(dataloader_val)

    print(f"{format_magnitude(train_size * config.context_length * config.batch_size)} train tokens")
    print(f"{format_magnitude(val_size * config.context_length * config.batch_size_val)} val tokens")

    return dataloader_train, dataloader_val

def main():
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    config = GPTConfig(vocab_size=tokenizer.eos_token_id+1)

    model, optimizer, loss_fn = get_train_obj(config)

    dataloaders, dataloader_val = get_dataloaders(config)

    print(f"{format_magnitude(model.num_parameters)} parameters")

    trainer = Trainer(
                model,
                optimizer,
                loss_fn,
                dataloaders,
                config=config,
                dataloader_val=dataloader_val,
                tokenizer=tokenizer)

    trainer.train()

if __name__ == "__main__":
    main()