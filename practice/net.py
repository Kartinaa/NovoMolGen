### Load pretrained model:
from transformers import AutoTokenizer, BertModel
import torch


### Define device:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pretrained = BertModel.from_pretrained("/home/yang2531/Documents/Project/NovoMolGen/cache/google-bert/bert-base-chinese/models--google-bert--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea").to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained("/home/yang2531/Documents/Project/NovoMolGen/cache/google-bert/bert-base-chinese/models--google-bert--bert-base-chinese/snapshots/8f23c25b06e129b6c986331a13d8d025a92cf0ea")
### Check the input demension and output demension:
# print(pretrained)


### Define downstream task:
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = torch.nn.Linear(768, 2)
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        # freeze the upstream model:
        with torch.no_grad():
            out = pretrained(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # get the last hidden state and train the downstream task:
        out = self.fc(out.last_hidden_state[:,0])
        out = out.softmax(dim=1)
        return out