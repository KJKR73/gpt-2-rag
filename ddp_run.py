import torch
import random
import pandas as pd
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        token_str = f"[user]\n{row['query']}\n"
        t_context = random.randint(a=0, b=4)
        for p in [f"passage_{i + 1}" for i in range(t_context)]:
            token_str += f"[context]\n{row[p]}\n"

        token_str += f"[response]\n{row['answer']}"

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
    
    
class DataRegistry(object):
    '''
    Define the Data Registry class
    '''
    def __init__(self, config, data_df):
        self.config = config
        self.data_df = data_df
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        special_tokens_dict = {'additional_special_tokens': ['[user]','[context]','[response]']}
        _ = self.tokenizer.add_special_tokens(special_tokens_dict)
        
    def get_loader_ddp(self):
        # Load the tokenize
        train_dataset = RAGDataset(self.data_df.sample(1000), tokenizer=self.tokenizer)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.config.BATCH_SIZE,
                                                   num_workers=self.config.NUM_WORKERS,
                                                   sampler=train_sampler)
        
        valid_dataset = RAGDataset(self.data_df.sample(1000), tokenizer=self.tokenizer)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=self.config.BATCH_SIZE,
                                                   num_workers=self.config.NUM_WORKERS)
        
        return train_loader, valid_loader, self.tokenizer
    
    def get_loader_amp(self):
        # Load the tokenize
        train_dataset = RAGDataset(self.data_df.sample(1000), tokenizer=self.tokenizer)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.config.BATCH_SIZE,
                                                   num_workers=self.config.NUM_WORKERS,
                                                   shuffle=True)
        
        valid_dataset = RAGDataset(self.data_df.sample(1000), tokenizer=self.tokenizer)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=self.config.BATCH_SIZE,
                                                   num_workers=self.config.NUM_WORKERS,
                                                   shuffle=True)
        
        return train_loader, valid_loader, self.tokenizer
    
    
def forward(model, data, config, rank):
    # Push the data to the device
    data = {k: v.to(rank) for k, v in data.items()}

    # Pass the data through the model
    if config.AMP:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(**data)
    else:
        output = model(**data)

    return output.loss, output.logits
    
def train_one_epoch(epoch, dataloader, scaler, optimizer, config, model, rank):
    # Put the model in training mode
    loss_meter = AverageMeter()
    model.train()

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for batch_idx, data in bar:
        # Zero the grads
        optimizer.zero_grad()

        # Forward pass
        loss, _ = forward(model, data, config, rank)
        loss_meter.update(loss.item(), data["input_ids"].shape[0])

        # Backward pass
        if config.AMP:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP)
            optimizer.step()

        # Set the description string
        bar.set_description(f"[TRAIN] || Epoch: {epoch + 1} | Batch : {batch_idx + 1} | Loss : {round(loss_meter.avg, ndigits=5)}")
        
    return loss_meter.avg

def validate_one_epoch(epoch, dataloader, config, model, rank):
    # Put the model in eval mode
    model.eval()
    
    # Iterative over the data
    loss_meter = AverageMeter()
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for _, data in bar:
        # Forward pass
        loss, _ = forward(model, data, config, rank)
        loss_meter.update(loss.item(), data["input_ids"].shape[0])

        # Set the description string
        bar.set_description(f"[VALID] || Epoch: {epoch + 1} | Loss : {round(loss_meter.avg, ndigits=5)}")

    return loss_meter.avg

def _ddp_main(gpu, ngpus_per_node, config):
    # Define the world size
    world_size = ngpus_per_node * config.NODES

    # Get the total device count
    ngpus_per_node = torch.cuda.device_count()
    rank = 0 * ngpus_per_node + gpu  
    dist.init_process_group(backend='nccl',
                            world_size=world_size,
                            rank=rank)
    
    # Print out some information
    print(f"Initilaized Process group | world_size : {world_size} | rank : {rank}")
    
    # Get the loaders
    data_df = pd.read_csv("/mnt/c/Ubuntu/interview_prep/gpt-2-rag/data/msmacro_train_data_rag.csv")    
    config.BATCH_SIZE = int(config.BATCH_SIZE / ngpus_per_node)
    config.NUM_WORKERS  = int(config.NUM_WORKERS / ngpus_per_node)
    loader_registry = DataRegistry(config, data_df)
    train_loader, valid_loader, tokenizer = loader_registry.get_loader_ddp()

    # Set the device to load to
    torch.cuda.set_device(gpu)
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    model.resize_token_embeddings(len(tokenizer))
    model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    if rank == 0:
        print(f"Model loaded in DDP model....")
        print(f"Using GPUS : {ngpus_per_node}")
        print(f"Batch size per GPU : {config.BATCH_SIZE}")
        print(f"Workers per GPU : {config.NUM_WORKERS}")
    
    # Load the optimizer
    optimizer = torch.optim.AdamW(params=model.parameters(),
                                  lr=config.LR)

    # Train the model
    print("Model training started.....")
    print(f"Using AMP : {config.AMP}")
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(config.EPOCHS):
        train_one_epoch(epoch, train_loader, scaler, optimizer, config, model, rank)
        
        if (rank == 0) :
            torch.save(model.state_dict(), config.PATH + f"{epoch + 1}.pt")
        
        with torch.no_grad():
            validate_one_epoch(epoch, valid_loader, config, model, rank)
            
    
def trainDDP():
    # Load the ddp
    n_gpu = torch.cuda.device_count()
    mp.spawn(_ddp_main,
             nprocs=n_gpu,
             args=(n_gpu, configDDP()))
    

class configDDP:
    LR = 5e-5
    NODES = 1 
    DDP = True
    AMP = True
    CLIP = 5.0
    EPOCHS = 3
    BATCH_SIZE = 2
    NUM_WORKERS = 6
    PATH = "/mnt/c/Ubuntu/interview_prep/gpt-2-rag/saved/weight_"


# MAIN
if __name__ == "__main__":
    # Train the model
    trainDDP()
