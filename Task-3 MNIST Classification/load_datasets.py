import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def load_mnist_data():
    # define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor()])
    # download and load the training data
    trainset = datasets.MNIST('../mnist_dataset', download=True, train=True, transform=transform)
    # trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    # download and load the test data
    testset = datasets.MNIST('../mnist_dataset', download=True, train=False, transform=transform)
    # testloader = DataLoader(testset, batch_size=64, shuffle=False)

    return trainset, testset