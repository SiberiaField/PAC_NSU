import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 3)
        self.out = torch.nn.Linear(3, 1)
        self.act = torch.nn.ReLU()
    
    def forward(self, x : torch.Tensor):
        h = self.fc(x)
        h = self.act(h)
        out = self.out(h)
        return self.act(out)
    
m = model()

x = torch.tensor([[0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 1.0, 1.0],
                  [1.0, 0.0, 0.0],
                  [1.0, 0.0, 1.0],
                  [1.0, 1.0, 0.0],
                  [1.0, 1.0, 1.0]])

y = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

l1_loss = torch.nn.L1Loss()

train_ds = TensorDataset(x, y)
train_dl = DataLoader(train_ds, 8)

opt = torch.optim.Adam(m.parameters(), lr=0.1)

epochs = 300
for epoch in range(epochs):
    for xb, yb in train_dl:
        pred = m(xb)
        loss = l1_loss(pred, yb)
        print(f'{epoch} {loss}')

        loss.backward()
        opt.step()
        opt.zero_grad()

print(m.forward(x))