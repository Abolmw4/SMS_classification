from model import MyModel
from data import *
from dataclasses import dataclass, field
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import preprocess_data

@dataclass
class TestSpam:
    device: str = field(default='cuda', repr=True)
    model: str = field(default_factory=str, repr=True)
    text: list = field(default_factory=list)
    label: list = field(default_factory=list)

    def __post_init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # if self.csv_file == '' or self.csv_file is None:
        #     raise ValueError("csv_file must be")
        # if self.model.find(".pth") != -1 or self.model.find(".pt") != -1:
        #     raise ValueError("input value must be .pth file or .pt file and must be absolute path!")

    def calculate_accuracy(self, y_pred, y, device) -> float:
        correct = (y_pred.argmax(1).to(device) == y.argmax(1).to(device)).type(torch.float).sum()
        acc = correct / y.shape[0]
        return acc

    def infrence(self):
        epoch_acc = 0.0
        Net = MyModel().to(self.device)
        Net.load_state_dict(torch.load(self.model))
        Net.eval()
        test_dataset = Data(text=self.text, lable=self.label)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Test", leave=True):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                y = torch.tensor(batch['label'], dtype=torch.float32)
                labels = y.to(self.device)
                outputs = Net(input_ids=input_ids, attention_mask=attention_mask)
                acc = self.calculate_accuracy(outputs, labels, self.device)
                epoch_acc += acc.item()
            return epoch_acc / len(test_dataloader)


if __name__ == "__main__":
    tx, lb = preprocess_data('/home/example/csv')
    s = TestSpam(device="cuda", text=tx, label=lb, model="/home/example.pt")
    print(s.infrence())
