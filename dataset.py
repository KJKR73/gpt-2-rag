import torch
from transformers import AutoTokenizer

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
    
    
class DataRegistry(object):
    '''
    Define the Data Registry class
    '''
    def __init__(self, config, data_df):
        self.config = config
        self.data_df = data_df
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def get_loader_ddp(self):
        # Load the tokenize
        train_dataset = RAGDataset(self.data_df.sample(100000), tokenizer=self.tokenizer)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.config.BATCH_SIZE,
                                                   num_workers=self.config.NUM_WORKERS,
                                                   sampler=train_sampler)
        
        valid_dataset = RAGDataset(self.data_df.sample(1000), tokenizer=self.tokenizer)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=self.config.BATCH_SIZE,
                                                   num_workers=self.config.NUM_WORKERS)
        
        return train_loader, valid_loader
    
    def get_loader_amp(self):
        # Load the tokenize
        train_dataset = RAGDataset(self.data_df.sample(100000), tokenizer=self.tokenizer)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=self.config.BATCH_SIZE,
                                                   num_workers=self.config.NUM_WORKERS,
                                                   shuffle=True)
        
        valid_dataset = RAGDataset(self.data_df.sample(1000), tokenizer=self.tokenizer)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=self.config.BATCH_SIZE,
                                                   num_workers=self.config.NUM_WORKERS,
                                                   shuffle=True)
        
        return train_loader, valid_loader