import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

class RAGDataset(object):
    '''
    Dataset for RAG
    '''
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        # Get the row from the dataframe
        row = self.data.iloc[index]

        # Define the chat template
        token_str = f"[user] {row['query']} \n"
        for p in [f"passage_{i + 1}" for i in range(2)]:
            token_str += f"[Context] \n {row[p]} \n \n"

        token_str += f"[Answer] \n {row['answer']}"

        # Encode the data
        tokenized_data = self.tokenizer.encode_plus(token_str,
                                                    return_tensors="pt",
                                                    truncation=True,
                                                    padding="max_length",
                                                    max_length=1024)

        return {
            "input_ids": tokenized_data["input_ids"].long(),
            "attention_mask": tokenized_data["attention_mask"].long(),
            "labels": tokenized_data["input_ids"].long(),
        }

    def __len__(self):
        return self.data.shape[0]
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class Trainer(object):
    '''
    Trainer to train the model on the Chat Dataset
    '''
    def __init__(self, config):
        # Define the instance variables
        self.config = config

        # Define the model and its utils
        self.model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(self.config.DEVICE)

        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config.LR)

    def forward(self, data):
        # Push the data to the device
        data = {k: v.to(self.config.DEVICE) for k, v in data.items()}

        # Pass the data through the model
        if self.config.AMP:
            with torch.autocast(device_type=self.config.DEVICE, dtype=torch.float16):
                output = self.model(**data)
        else:
            output = self.model(**data)
        
        return output.loss, output.logits

    def train_one_epoch(self, epoch, dataloader, scaler):
        # Put the model in training mode
        loss_meter = AverageMeter()
        self.model.train()

        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, data in bar:
            # Zero the grads
            self.optimizer.zero_grad()
            
            # Forward pass
            loss, _ = self.forward(data)
            loss_meter.update(loss.item(), data["input_ids"].shape[0])
            
            # Backward pass
            if self.config.AMP:
                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.CLIP)
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.CLIP)
                self.optimizer.step()

            # Set the description string
            bar.set_description(f"Epoch: {epoch + 1} | Batch : {batch_idx + 1} | Loss : {round(loss_meter.avg, ndigits=5)}")

        return loss_meter.avg

    def train(self, dataloader):
        # Define the scaler
        scaler = torch.cuda.amp.GradScaler()
        
        # Loop over epochs and train
        for epoch in range(self.config.EPOCHS):
            self.train_one_epoch(epoch, dataloader, scaler)      
            
class config:
    DEVICE = "cuda"
    LR = 5e-4
    EPOCHS = 10
    AMP = True
    CLIP = 5.0 

# Load the data
data_df = pd.read_csv("./data/msmacro_train_data_rag.csv")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
train_dataset = RAGDataset(data_df, tokenizer=tokenizer)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, num_workers=6, shuffle=True)

# Train the model
trainer = Trainer(config())
trainer.train(train_loader)
