import torch
from torch.utils import data
from torchvision import transforms, datasets


def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    # 使用PyTorch内置的备用下载源，避免官方源不稳定
    mnist_train = datasets.FashionMNIST(
        root="./data",
        train=True,
        transform=trans,
        download=True
    )
    mnist_test = datasets.FashionMNIST(
        root="./data",
        train=False,
        transform=trans,
        download=True
    )

    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=0)
    test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=0)
    return train_iter, test_iter