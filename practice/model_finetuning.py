from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

class Mydataset(Dataset):
    def __init__(self, split):
        self.dataset = load_from_disk("/home/yang2531/Documents/Project/NovoMolGen/cache/lansinuote/ChnSentiCorp")
        if split == "train":
            self.dataset = self.dataset["train"]
        elif split == "validation":
            self.dataset = self.dataset["validation"]
        elif split == "test":
            self.dataset = self.dataset["test"]
        else:
            raise ValueError(f"Invalid split: {split}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        text = self.dataset[index]["text"]
        label = self.dataset[index]["label"]
        return text, label
    
    
### We need to test the dataset:
if __name__ == "__main__":
    dataset = Mydataset(split="validation")
    for data in dataset:
        print(data)
        