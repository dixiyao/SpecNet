import torch
import torch.nn as nn
from layers import FourierConv2D, SpectralPooling2d,FFT,IFFT
import layers_org

class CNNMNIST(nn.Module):
    def __init__(self):
        super(CNNMNIST, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3,padding=1,bias=False),
            nn.Tanh(),
            nn.Conv2d(32, 64, 3,padding=1,bias=False),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(12544, 128),
            nn.ReLU(True),
            nn.Linear(128, 10),
            nn.LogSoftmax(1)
        )
        self.name = 'VanillaCNN'

    def forward(self, x):
        return self.net(x)


class FCNNMNIST(nn.Module):
    def __init__(self):
        super(FCNNMNIST, self).__init__()

        self.net = nn.Sequential(
            FFT(),
            FourierConv2D(1, 32, 3,bias=False),
            nn.Tanh(),
            FourierConv2D(32, 64, 3,bias=False),
            SpectralPooling2d(2),
            IFFT(),
            nn.Flatten(),
            nn.Linear(14*14*64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(1)
        )
        self.name = 'FourierCNN'

    def forward(self, x):
        return self.net(x)


def weight_init(net):
    for module in net.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            torch.nn.init.uniform_(module.weight, -0.1, 0.1)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    x = torch.zeros([4, 1, 28, 28]).to(device)
    cnn = CNNMNIST().to(device)
    fcnn = FCNNMNIST().to(device)
    print(cnn(x).size())
    print(fcnn(x).size())