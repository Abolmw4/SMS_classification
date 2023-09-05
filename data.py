from torch.utils.data import Dataset
import torch


class Data(Dataset):
    def __init__(self, text, lable):
        super(Data, self).__init__()
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer',
                                        'bert-base-cased-finetuned-mrpc')
        self.texts = text
        self.lables = lable

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        lable = self.lables[index]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=64, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(),
                'label': lable.reshape(-1)}
