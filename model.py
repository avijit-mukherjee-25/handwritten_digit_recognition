import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import evaluate
from torch.optim import AdamW
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pytorch
from mlflow.models.signature import infer_signature


# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


batch_size = 32
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-4
num_epochs = 5


def get_data():

    train_dataset = datasets.MNIST(root="./datasets/", train=True, download=True, transform=ToTensor())
    test_dataset = datasets.MNIST(root="./datasets/", train=False, download=True, transform=ToTensor())

    # nornalize the data
    imgs = torch.stack([img for img, _ in train_dataset], dim=0)
    print (imgs.shape)
    mean = imgs.view(1, -1).mean(dim=1)
    std = imgs.view(1, -1).std(dim=1)
    print (mean, std)

    # transform data
    mnist_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    train_dataset = datasets.MNIST(root="./datasets/", train=True, download=False, transform=mnist_transforms)
    test_dataset = datasets.MNIST(root="./datasets/", train=False, download=False, transform=mnist_transforms)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader


def train(model, optimizer, loss_fn, train_dataloader):
    train_loss = 0.0
    model.train()
    metric = evaluate.load("accuracy")
    for step, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        train_loss += loss.item()
        metric.add_batch(predictions=logits.argmax(dim=1), references=y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step%1000==0:
            print (f'training loss at {step} is {train_loss}')
    train_accuracy = metric.compute()
    return train_loss, train_accuracy

@torch.no_grad()
def eval(model, test_dataloader):
    model.eval()
    metric = evaluate.load("accuracy")
    test_loss = 0.0
    for _, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = loss_fn(logits, y)
        test_loss += loss.item()
        metric.add_batch(predictions=logits.argmax(dim=1), references=y)
    test_accuracy = metric.compute()
    model.train()
    return test_loss, test_accuracy

# define the model
class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel (black & white), 6 output channels, 5x5 square convolution
        # kernel
        self.feature = nn.Sequential(
            #1
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),   # 28*28->32*32-->28*28
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 14*14

            #2
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),  # 10*10
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 5*5

        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=10),
        )

    def forward(self, x):
        return self.classifier(self.feature(x))

def main():
    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    print("Training MLflow Tracking URI:", mlflow.get_tracking_uri())

    mlflow.set_experiment('lenet_mnist_experiment')

    with mlflow.start_run():
        LeNet_model = LeNet() 
        LeNet_model.to(device)
        optimizer = AdamW(LeNet_model.parameters(), lr=learning_rate)
        train_dataloader, test_dataloader = get_data()
        mlflow.log_params(
            {
                'batch_size': batch_size,
                'loss_function': loss_fn,
                'learning_rate': learning_rate,
                'num_epochs': num_epochs
            }
        )

        for epoch in range(num_epochs):
            print (f'Epoch: {epoch}')
            train_loss, train_accuracy = train(LeNet_model, optimizer, loss_fn, train_dataloader)
            print (f'train loss at epoch {epoch} is {train_loss}; train accuracy is {train_accuracy}')
            test_loss, test_accuracy = eval(LeNet_model, test_dataloader)
            print (f'test loss at epoch {epoch} is {test_loss}; test accuracy is {test_accuracy}')
            mlflow.log_metric('train_accuracy', train_accuracy['accuracy'], step=epoch)
            mlflow.log_metric('test_accuracy', test_accuracy['accuracy'], step=epoch)
    
        # Log the model
        # Sample input: use one sample from the test set
        sample_input, sample_target = next(iter(test_dataloader))

        # Flatten input for serving (since serving usually expects flattened input)
        flattened_input = sample_input.view(sample_input.size(0), -1).numpy()

        # Get output from model
        with torch.no_grad():
            sample_output = LeNet_model(sample_input).numpy()

        # Infer signature
        signature = infer_signature(flattened_input, sample_output)

        mlflow.pytorch.log_model(
            pytorch_model=LeNet_model, 
            artifact_path='lenet_model',
            signature=signature,
            registered_model_name='lenet_model'
            )
        
if __name__ == "__main__":
    main()