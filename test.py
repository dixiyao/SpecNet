import torch
from torch.autograd import Variable

import numpy as np

import DataLoaders
import models
import utils
import vgg

def test(epoch,model,test_loader,device):
    model.eval()  # 让模型变为测试模式，主要是保证dropout和BN和训练过程一致。BN是指batch normalization
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]  # 获得得分最高的类别
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(
         correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

def main():
    test_loader = DataLoaders.get_loader('mnist', '../data', 512, 'test')

    if torch.cuda.is_available():
        device = torch.device('cuda',0)
    else:
        device = torch.device('cpu')

    CNN=models.CNNMNIST()
    state=torch.load('./checkpoints/CNNMNIST.plt')
    CNN.load_state_dict(state['net'])
    CNN=CNN.to(device)
    test(0,CNN,test_loader,device)
    FCN=models.FCNNMNIST()
    utils.FCN_load_parameter(FCN,CNN)
    FCN=FCN.to(device)
    test(0, FCN, test_loader, device)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
