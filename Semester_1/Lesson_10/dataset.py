import numpy as np
import torchvision

def encode_label(i):
    # 5 -> [[0], [0], [0], [0], [0], [1], [0], [0], [0], [0]]
    e = np.zeros((10, 1))
    e[i] = 1
    return e

def shape_data(data):
    features = [np.reshape(x[0][0].numpy(), (784, 1)) for x in data]
    labels = [encode_label(y[1]) for y in data]
    return list(zip(features, labels))

def get_dataset():
    # Initializing the transform for the dataset
    transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5), (0.5))])

    # Downloading the MNIST dataset
    train_dataset = torchvision.datasets.MNIST(
        root="./MNIST/train", train=True,
        transform=torchvision.transforms.ToTensor(),
        download=False)

    test_dataset = torchvision.datasets.MNIST(
        root="./MNIST/test", train=False,
        transform=torchvision.transforms.ToTensor(),
        download=False)

    train = shape_data(train_dataset)
    test = shape_data(test_dataset)
    return train, test