from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
import re
import string
import nltk
import pandas as pd
import torch

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def calculate_accuracy(y_pred, y, device):
    correct = (y_pred.argmax(1).to(device) == y.argmax(1).to(device)).type(torch.float).sum()
    acc = correct / y.shape[0]
    return acc


def train(model, data_loader, optimizer, criterion, device):
    epoch_loss = 0.0
    epoch_acc = 0.0
    model.train()
    for batch in tqdm(data_loader, desc="Training", leave=True):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y = torch.tensor(batch['label'], dtype=torch.float32)
        labels = y.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        l2_norm = sum(p.pow(2.0).sum()
                      for p in model.parameters())
        loss = criterion(outputs, labels) + l2_norm * 0.0008
        acc = calculate_accuracy(outputs, labels, device)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader), epoch_acc / len(data_loader)


def evaluate(model, data_loader, criterion, device):
    epoch_loss = 0.0
    epoch_acc = 0.0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluate", leave=True):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y = torch.tensor(batch['label'], dtype=torch.float32)
            labels = y.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            l2_norm = sum(p.pow(2.0).sum()
                          for p in model.parameters())
            loss = criterion(outputs, labels) + l2_norm * 0.0008
            epoch_loss += loss.item()
            acc = calculate_accuracy(outputs, labels, device)
            epoch_acc += acc.item()
        return epoch_loss / len(data_loader), epoch_acc / len(data_loader)


def preprocess(text):
    text = re.sub(r'https?://\S+|www.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z]+', ' ', text)
    text = re.sub(r'[0-9]', '', text)
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    return text


def preprocess_data(dataset):
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit([["spam", "ham"]])

    data = pd.read_csv(dataset)
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    data.rename(columns={"v1": "Category", "v2": "Message"}, inplace=True)
    data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1, inplace=True)
    data['Message'] = data.Message.map(preprocess)
    for item in range(len(data)):
        if data['Category'].iloc[item] == 'spam':
            data['Category'].iloc[item] = enc.transform([['spam', 0]]).toarray()
        else:
            data['Category'].iloc[item] = enc.transform([[0, 'ham']]).toarray()
    texts = data['Message'].tolist()
    labels = data['Category'].tolist()
    return texts, labels
