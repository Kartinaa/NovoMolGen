import torch
from model_finetuning import Mydataset
from net import Model
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCH = 100

tokenizer = BertTokenizer.from_pretrained("/home/yang2531/Documents/Project/NovoMolGen/cache/google-bert/bert-base-chinese/models--google-bert--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea")

# Define collate function to encode the text before inputting into the model:
def collate_fn(batch):
    '''
    We not do pre-encoding because it is too slow (But why?).
    '''
    texts = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    # Encode the text:
    encodings = tokenizer.batch_encode_plus(texts, 
                                            padding="max_length", 
                                            max_length=350,
                                            add_special_tokens=True,
                                            truncation=True, 
                                            return_tensors="pt",
                                            return_length=True)
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    token_type_ids = encodings['token_type_ids']
    labels = torch.LongTensor(labels)
    return input_ids, attention_mask, token_type_ids, labels


# Create dataloader:
training_dataset = Mydataset(split="train")
training_dataloader = DataLoader(training_dataset, 
                                 batch_size=32, 
                                 shuffle=True,
                                 drop_last = True, # drop the last batch if the length is not divisible by the batch size
                                 collate_fn=collate_fn
                                 )

# Start training:

if __name__ == "__main__":
    
    print(DEVICE)
    model = Model().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=5e-4)
    loss_func = torch.nn.CrossEntropyLoss()
    
    
    model.train()
    for epoch in range(EPOCH):
        for i, batch in enumerate(training_dataloader):
            # put data to GPU:
            input_ids, attention_mask, token_type_ids, labels = batch
            input_ids = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            token_type_ids = token_type_ids.to(DEVICE)
            labels = labels.to(DEVICE)
            
            # forward pass:
            outputs = model(input_ids, attention_mask, token_type_ids)
            
            loss = loss_func(outputs, labels)
            
            # Reset the gradient:
            optimizer.zero_grad()
            
            # backward pass:
            loss.backward()
            
            # update the parameters:
            optimizer.step()
            
            if i%5 == 0:
                out = outputs.argmax(dim=1)
                acc = (out == labels).sum() / len(labels)
                print(f"Epoch {epoch} | Step {i} | Loss {loss.item()} | Accuracy {acc.item()}")
        
        # save the model:
        torch.save(model.state_dict(), f"/home/yang2531/Documents/Project/NovoMolGen/practice/params/model_{epoch}.pth")
        # In case `torch.save` raise error during saving.
        print(epoch)