

import torch
from torch import nn 
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import random

from pathlib import Path


MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'OCR_CNN_MODEL.pth'
SAVE_PATH = MODEL_PATH / MODEL_NAME

# Device agnostic code >> sets 'device' to GPU if nVIDIA GPU is available, otherwise it is set to CPU
device =  'cuda' if torch.cuda.is_available() else 'cpu'

# Download EMNIST handritten datasets for training and testing (28x28 resolution)
train_data = datasets.EMNIST(root='data', train=True, download=True, transform=ToTensor(), split='letters')
test_data = datasets.EMNIST(root='data', train=False, download=True, transform=ToTensor(), split='letters')

# Test to see if the dataset is downloaded and accessible
#image, label = train_data[0]
#plt.imshow(image.squeeze(), cmap='gray')
#plt.title(chr(96+label))
#plt.show()

########## HYPERPARAMETERS ############
batchSize = 32
epochs = 100
learningRate = 0.07

#######################################

# Create Dataloaders from the train/test datasets in order to easily feed to NN when training/testing
train_dataloader = DataLoader(train_data, batch_size=batchSize, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batchSize, shuffle=False)


# character recognition CNN class
class ocrModel(nn.Module):

    def __init__(self, inputShape, hiddenUnits, outputShape):   # 3 arguments: 
                                                                # inputShape == number of channels, 
                                                                # hiddenUnits == # of neurons in hidden layer,
                                                                # outputShape == # of classes
        super().__init__()
        self.layers1 = nn.Sequential(               # Model consists of 3 nn.Sequentials. First 2 are Convolutional Layers
            nn.Conv2d(                              # 3rd nn.Sequential is a classifier layer that uses nn.Flatten and a nn.Linear for output
                in_channels=inputShape, 
                out_channels=hiddenUnits, 
                kernel_size=3,
                stride=1,
                padding=1), 
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hiddenUnits,
                out_channels=hiddenUnits, 
                kernel_size=3,
                stride=1, 
                padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layers2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hiddenUnits, 
                out_channels=hiddenUnits, 
                kernel_size=3,
                stride=1,
                padding=1), 
                nn.ReLU(),
            nn.Conv2d(
                in_channels=hiddenUnits,
                out_channels=hiddenUnits, 
                kernel_size=3,
                stride=1, 
                padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(2))
        self.classifiers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hiddenUnits*49,
                out_features=outputShape)
        )


    def forward(self, x):               # Forward pass
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.classifiers(x)
        return x


# Instantiate the character recogntion model >> 1 input channel, 10 neurons/hidden layer, num of classes
if Path.exists(SAVE_PATH):
    model0 = ocrModel(1, 10, len(train_data.classes))
    model0.load_state_dict(state_dict=torch.load(SAVE_PATH))
    model0.to(device)
    print('Model Loaded')
else:
    model0 = ocrModel(1, 10, len(train_data.classes))
    model0.to(device)   # Send model to device

# Create loss function and optimizer (I could be wrong, but CrossEntropyLoss and SGD are pretty much standard and are not changed)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model0.parameters(), lr=learningRate)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc


def train(dataloader, model, loss_fn, optimizer, device):

    for batch, (X, Y) in enumerate(dataloader):

        X, Y = X.to(device), Y.to(device)

        model.train()

        train_pred = model(X)

        loss = loss_fn(train_pred, Y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


def test(dataloader, model, accuracy_fn, loss_fn, device):

    test_loss, test_accuracy = 0, 0
    model.eval()

    with torch.inference_mode():
        for X, Y in dataloader:

            X, Y = X.to(device), Y.to(device)
            
            test_pred = model(X)

            test_loss += loss_fn(test_pred, Y)
            test_accuracy += accuracy_fn(Y, test_pred.argmax(dim=1))

        test_loss /= len(dataloader)
        test_accuracy /= len(dataloader)

    print(f"Test loss:  {test_loss:.5f} | Test accuracy:  {test_accuracy:.2f}%\n")

def evaluate(sample, model, device):
    with torch.inference_mode():
        
        predList = []
        truth = []
        for image, label in random.sample(list(sample), k=10):
            truth.append(label)
            image = torch.unsqueeze(torch.Tensor(image), dim=0).to(device)
            predlogits = model(image)
            pred = torch.softmax(predlogits.squeeze(), dim=0).argmax()
            predList.append(pred.cpu().item())
        print(f"Truth: {truth}, Guess: {predList}")


for epoch in tqdm(range(epochs)):
    print(f"\n\nTrain Epoch: {epoch+1}\n----------")
    train(train_dataloader, model0, loss_fn, optimizer, device)
    test(test_dataloader, model0, accuracy_fn, loss_fn, device)
evaluate(test_data, model0, device)

torch.save(obj=model0.state_dict(), f=SAVE_PATH)
print('\nModel Saved')