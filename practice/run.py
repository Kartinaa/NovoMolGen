import torch
from net import Model
from transformers import BertTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

names = ["negative comment", "positive comment"]
model = Model().to(DEVICE)

tokenizer = BertTokenizer.from_pretrained("/home/yang2531/Documents/Project/NovoMolGen/cache/google-bert/bert-base-chinese/models--google-bert--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea")

def collate_fn(user_input):
    sents = []
    sents.append(user_input)
    
    data = tokenizer.batch_encode_plus(sents, 
                                       padding="max_length", 
                                       max_length=350, 
                                       add_special_tokens=True, 
                                       truncation=True, 
                                       return_tensors="pt", 
                                       return_length=True)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    return input_ids, attention_mask, token_type_ids


def test():
    model.load_state_dict(torch.load(f"/home/yang2531/Documents/Project/NovoMolGen/practice/params/model_3.pth"))
    model.eval()
    while True:
        user_input = input("Enter a sentence: ")
        if user_input == "q":
            break
        input_ids, attention_mask, token_type_ids = collate_fn(user_input)
        input_ids = input_ids.to(DEVICE)
        attention_mask = attention_mask.to(DEVICE)
        token_type_ids = token_type_ids.to(DEVICE)
        out = model(input_ids, attention_mask, token_type_ids)
        out = out.argmax(dim=1)
        print(names[out])
        print(out)

if __name__ == "__main__":
    test()