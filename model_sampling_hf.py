import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("bisectgroup/NovoMolGen_32M_SAFE_BPE", revision='main', device_map='auto')
tokenizer = AutoTokenizer.from_pretrained("bisectgroup/NovoMolGen_32M_SAFE_BPE", revision='main')

# Define the input ids for generation
input_ids = torch.tensor([[tokenizer.bos_token_id]]).expand(4, -1).contiguous().to(model.device)


outs = model.generate(input_ids=input_ids, 
                      temperature=1.0,
                      max_length=128, 
                      do_sample=True, 
                      pad_token_id=tokenizer.pad_token_id,
                      top_k=50,
                      top_p=0.95)
print(outs)
molecules = [t.replace(" ", "") for t in tokenizer.batch_decode(outs, skip_special_tokens=True)]
print(molecules)