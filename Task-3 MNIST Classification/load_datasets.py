import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def load_mnist():
    # define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor()])
    # download and load the training data
    trainset = datasets.MNIST('../datasets', download=True, train=True, transform=transform)
    # trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    # download and load the test data
    testset = datasets.MNIST('../datasets', download=True, train=False, transform=transform)
    # testloader = DataLoader(testset, batch_size=64, shuffle=False)

    return trainset, testset


def load_fasion_mnist():
    # define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor()])
    # download and load the training data
    trainset = datasets.FashionMNIST('../datasets', download=True, train=True, transform=transform)
    # trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    # download and load the test data
    testset = datasets.FashionMNIST('../datasets', download=True, train=False, transform=transform)
    # testloader = DataLoader(testset, batch_size=64, shuffle=False)

    return trainset, testset


def load_cifar10():
    # define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor()])
    # download and load the training data
    trainset = datasets.CIFAR10('../datasets', download=True, train=True, transform=transform)
    # trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    # download and load the test data
    testset = datasets.CIFAR10('../datasets', download=True, train=False, transform=transform)
    # testloader = DataLoader(testset, batch_size=64, shuffle=False)

    return trainset, testset