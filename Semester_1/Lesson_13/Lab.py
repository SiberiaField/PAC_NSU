from pathlib import Path
import requests
import pickle
import gzip
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

x_train, x_valid = map(torch.tensor, (x_train, x_valid))
y_train = torch.tensor(y_train, dtype=int)
y_valid = torch.tensor(y_valid, dtype=int)

def encode_tensor(t : torch.Tensor) -> torch.Tensor:
    res = torch.empty((t.size(0), 10))
    for i in range(t.size(0)):
        res[i] = torch.zeros(10)
        res[i][t[i]] = 1
    return res

y_train = encode_tensor(y_train)
y_valid = encode_tensor(y_valid)

bs = 64
lr = 0.001
epochs = 5

l1_loss = torch.nn.L1Loss()

train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

model = torch.nn.Sequential(torch.nn.Linear(784, 256),
                            torch.nn.ReLU(),
                            torch.nn.Linear(256, 64),
                            torch.nn.ReLU(),
                            torch.nn.Linear(64, 10),
                            torch.nn.Softmax(0))

opt = torch.optim.Adam(model.parameters(), lr=lr)

confusion_matrix = np.zeros((10, 10), dtype=np.int64)
for epoch in range(epochs):
    model.train()
    for xb, yb in train_dl:
        pred = model(xb)
        loss = l1_loss(pred, yb)

        for i in range(yb.size(0)):
            actual_label = torch.argmax(yb[i])
            predicted_label = torch.argmax(pred[i])
            confusion_matrix[actual_label][predicted_label] += 1

        loss.backward()
        opt.step()
        opt.zero_grad()
    
    model.eval()
    with torch.no_grad():
        valid_loss = sum(l1_loss(model(xb), yb) for xb, yb in valid_dl)
    
    print(f'epoch {epoch}')
    print(f'avg loss {valid_loss / len(valid_dl)}')

    TP = np.diag(confusion_matrix)
    FP = np.sum(confusion_matrix - np.diag(TP), 0)
    FN = np.sum(confusion_matrix - np.diag(TP), 1)

    res = {"precision" : [], "recall" : []}
    for digit in range(10):
        precision = TP[digit] / (TP[digit] + FP[digit])
        recall = TP[digit] / (TP[digit] + FN[digit])
        precision = np.round(precision, 2) * 100
        recall = np.round(recall, 2) * 100

        res["precision"].append(precision)
        res["recall"].append(recall)
    print('precision ')
    df = pd.DataFrame(res)
    print(df, '\n')