import torch

l1_loss = torch.nn.L1Loss()

# Task_1
model_1 = torch.nn.Sequential(torch.nn.Linear(64, 32), 
                            torch.nn.ReLU(), 
                            torch.nn.Linear(32, 10, bias=False),
                            torch.nn.ReLU())

x = torch.rand(4, 64) # batch size = 4
y = torch.rand(4, 10)

y_pred = model_1.forward(x)

loss = l1_loss(y, y_pred)
print('Loss 1:', loss)

# Task_2

class model_2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(256, 64)
        self.act1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 16)
        self.act2 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(16, 4, bias=False)
        self.act3 = torch.nn.Softmax(1)
    
    def forward(self, x):
        h = self.fc1(x)
        h = self.act1(h)
        h = self.fc2(h)
        h = self.act2(h)
        h = self.fc3(h)
        return self.act3(h)

x = torch.rand(4, 256) # batch size = 4
y = torch.rand(4, 4)

print(x.shape)

m_2 = model_2()
y_pred = m_2.forward(x)

loss = l1_loss(y, y_pred)
print('Loss 2:', loss)

# Task_3
class model_3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 2)
        self.conv2 = torch.nn.Conv2d(8, 16, 2)
        self.act = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2)
    
    def forward(self, x):
        h = self.conv1(x)
        h = self.pool(h)
        h = self.act(h)
        h = self.conv2(h)
        h = self.pool(h)
        return self.act(h)

m_3 = model_3()
x = torch.rand(4, 3, 19, 19) # batch size = 4
print('Task 3:', m_3.forward(x).shape)

# Task_4
batch_size = 4
x = torch.rand(batch_size, 3, 19, 19)
output = m_3.forward(x).view((batch_size, 256))
print('Task 4:', m_2.forward(output).shape)