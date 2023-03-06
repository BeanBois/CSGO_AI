#adapted from tutorials from
#https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#download training data
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)


#download test data 
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_sz = 64

train_dataloader = DataLoader(training_data, batch_size=batch_sz)
test_dataloader = DataLoader(test_data, batch_size=batch_sz)

for X,y in test_dataloader:
    print("Shape of X [N,C,H,W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_reu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_reu_stack(x)
        return logits

    
model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizers = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizers):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        #Computer prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        #Backward propagation
        optimizers.zero_grad()
        loss.backward()
        optimizers.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    test_loss, corrrect = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%  , Avg loss: {test_loss:>8f} \n")
    
    
#Hyperparameters
#NO OF EPOCH: the number of times to iterate through dataset
#BATCH SIZE: the number of data samples propagated thhrough the entwork before the parameters are updated
#LEARNING RATE: how much to update the model parameters at each batch/epoch


epochs = 5 
#each iteration of the optimisation loop is called an epoch
#2 parts: Train loop -- iterate through data set and converge to optimal parameters
#         Validation/Test loop -- iterate over test data to check model performance and see if its improving/overfitting

#LOSS FUNCTION: measures how well the model's output matches the target
#Loss function: measures how well the model's output matches the target
#loss function measures the degree of dissimilarity of obtained results to the target values, and it is the loss function
#that we want to minimize during training
#calculation done by comparing predictions and true values
#Common function
    #nn.MSELoss: mean squared error for regression
    #nn.NLLLoss: negative log likelihood for classification
    #nn.CrossEntropyLoss: combines nn.LogSoftmax() and nn.NLLLoss() in one single class


#OPTIMIZER: algorithm used to update the model's parameters
#optimisation is the process of adjusting model parameters to reduce model error in each training step
#optimzer specify how this is done.
optimizers = torch.optim.SGD(model.parameters(), lr=1e-3)
#in training loop, optimisation happends in 3 step,
    # call optimiser.zero_grad() to reset the gradient of model para. Gradients by default add up, and we dont want to snowball it
    #backpropagate the prediction loss with a call to loss.backward(). Pytorch stores the gradients for each model parameter
    #once gradients received, we call optimiser.step() to adjust the parameters by the gradients collected in backward pass

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizers)
    test(test_dataloader, model, loss_fn)
print("Done!")