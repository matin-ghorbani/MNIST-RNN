import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

INPUT_SIZE = 28
SEQUENCE_LENGTH = 28
NUM_LAYERS = 2
HIDDEN_SIZE = 256
NUM_CLASSES = 10
LR = .001
BATCH_SIZE = 64
EPOCHS = 3


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # N, sequence, features
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.lin = nn.Linear(hidden_size * SEQUENCE_LENGTH, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(device)  # Hidden state

        # Forward Prop
        x, _ = self.rnn(x, h0)  # _: new hidden state
        x = x.reshape(x.shape[0], -1)

        return self.lin(x)

train_dataset = MNIST('./data', transform=transforms.ToTensor(), download=True)
test_dataset = MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=True)

model = RNN(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)

optimizer = optim.Adam(model.parameters(), LR)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    for idx, batch  in enumerate(train_loader):
        x, y = map(lambda x: x.to(device), batch)
        x = x.squeeze(1)

        y_pred = model(x)

        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def check_accuracy(loader: DataLoader, model: nn.Module):
    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')
    
    num_correct = 0
    num_samples = 0

    model.eval()

    with torch.no_grad():
        for batch in loader:
            x, y = map(lambda x: x.to(device), batch)
            x = x.squeeze(1)

            y_pred = model(x)
            _, predictions = y_pred.max(1)

            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
