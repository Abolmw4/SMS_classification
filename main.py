import os.path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
from data import *
import torch.nn as nn
from model import MyModel
import matplotlib.pyplot as plt
import numpy as np
import argparse
from utils import *


def main(args):
    tex, lab = preprocess_data(args.input)
    t_texts, test_texts, t_labels, test_labels = train_test_split(tex, lab, test_size=0.05, random_state=42)
    train_texts, val_texts, train_labels, val_labels = train_test_split(t_texts, t_labels, test_size=0.1,
                                                                        random_state=42)
    Train_acc, Val_acc = list(), list()
    Train_loss, Val_loss = list(), list()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    myNet = MyModel().to(device)
    optimizer = optim.Adam(myNet.parameters(), lr=args.learningrae)

    weight = torch.tensor((0.7, 0.3)).to(device)
    criterion = nn.BCELoss(weight=weight)
    number_of_epochs = args.epoch
    batch_size = args.Batchsize

    train_dataset = Data(train_texts, train_labels)
    val_dataset = Data(val_texts, val_labels)
    test_dataset = Data(test_texts, test_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    for epoch in range(number_of_epochs):
        train_loss, train_acc = train(myNet, train_dataloader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(myNet, val_dataloader, criterion, device)

        print(f'Epoch: {epoch + 1:02}')
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"\tVal loss: {valid_loss:.3f} | val Acc: {valid_acc * 100:.2f}%")

        Train_acc.append(train_acc * 100)
        Val_acc.append(valid_acc * 100)
        Train_loss.append(train_loss)
        Val_loss.append(valid_loss)

        torch.save(myNet.state_dict(), os.path.join(args.weight, f'weight{epoch + 1}.pt'))

        epoch_acc = 0.0
        Net = MyModel().to("cuda")
        Net.load_state_dict(torch.load(os.path.join(args.weight, f'weight{epoch + 1}.pt')))
        Net.eval()
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Test", leave=True):
                input_ids = batch['input_ids'].to('cuda')
                attention_mask = batch['attention_mask'].to('cuda')
                y = torch.tensor(batch['label'], dtype=torch.float32)
                labels = y.to('cuda')
                outputs = Net(input_ids=input_ids, attention_mask=attention_mask)
                acc = calculate_accuracy(outputs, labels, 'cuda')
                epoch_acc += acc.item()
            print(f'test_acc: {epoch_acc / len(test_dataloader)}')

    x = [i + 1 for i in range(number_of_epochs)]
    plt.plot(np.array(x), np.array(Train_acc), label='train')
    plt.plot(np.array(x), np.array(Val_acc), label='val')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
    plt.plot(np.array(x), np.array(Train_loss), label='train')
    plt.plot(np.array(x), np.array(Val_loss), label='val')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="spam calssification")
    parser.add_argument('-i', '--input', default='/home/example.csv',
                        help="Input CSV file absolute path")
    parser.add_argument('-o', '--weight', default="/home/", help="director for save weights")
    parser.add_argument('-b', '--Batchsize', default=32, help="batch size")
    parser.add_argument('-e', '--epoch', default=10, help="number of epoch for train")
    parser.add_argument('-lr', '--learningrae', default=0.0001)
    args = parser.parse_args()
    main(args)
