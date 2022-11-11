

import torch
from torch import nn 
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt


device =  'cuda' if torch.cuda.is_available() else 'cpu'

train_data = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
test_data = datasets.MNIST(root='data', train=False, download=True, transform=ToTensor())

#plt.imshow(image.squeeze(), cmap='gray')
#plt.show()

########## HYPERPARAMETERS ############

batchSize = 32
epochs = 50
learningRate = 0.05

#######################################

train_dataloader = DataLoader(train_data, batch_size=batchSize, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batchSize, shuffle=False)

class ocrModel(nn.Module):
    def __init__(self, inputShape, hiddenUnits, outputShape):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=inputShape, out_features=hiddenUnits),
            nn.ReLU(), 
            nn.Linear(in_features=hiddenUnits, out_features=outputShape),
            nn.ReLU()
            )

    def forward(self, x):
        return self.layers(x)

model0 = ocrModel(784, 20, len(train_data.classes))
model0.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model0.parameters(), lr=learningRate)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc


def train(dataloader, model, accuracy_fn, loss_fn, optimizer, device):

    train_loss, train_accuracy = 0, 0

    for batch, (X, Y) in enumerate(train_dataloader):

        X, Y = X.to(device), Y.to(device)

        model.train()

        train_pred = model(X)

        loss = loss_fn(train_pred, Y)
        train_loss += loss
        train_accuracy += accuracy_fn(Y, train_pred.argmax(dim=1))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    train_loss /= len(dataloader)
    train_accuracy /= len(dataloader)
    
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_accuracy:.2f}\n\n")


def test(dataloader, model, accuracy_fn, loss_fn, device):

    test_loss, test_accuracy = 0, 0

    with torch.inference_mode():
        for X, Y in dataloader:

            X, Y = X.to(device), Y.to(device)
            
            test_pred = model(X)

            test_loss += loss_fn(test_pred, Y)
            test_accuracy += accuracy_fn(Y, test_pred.argmax(dim=1))

        test_loss /= len(dataloader)
        test_accuracy /= len(dataloader)

    print(f"\nTest loss: {test_loss:.5f}, Test acc: {test_accuracy:.2f}%\n")


for epoch in tqdm(range(epochs)):
    print(f"\n\nTrain Epoch: {epoch+1}\n----------")
    train(train_dataloader, model0, accuracy_fn, loss_fn, optimizer, device)
