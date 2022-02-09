import torch
from torch.autograd import Variable

import numpy as np

import DataLoaders
import models
import vgg

def train(epoch,model,train_loader,optimizer,criterion,device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()  # 相当于更新权重值
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    return loss.item()

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
    train_loader=DataLoaders.get_loader('mnist','../data',512,'train')
    valid_loader = DataLoaders.get_loader('mnist', '../data', 512, 'valid')

    if torch.cuda.is_available():
        device = torch.device('cuda',0)
    else:
        device = torch.device('cpu')

    model=models.FCNNMNIST()
    #model=vgg.VGG_11()
    model=model.to(device)
    criterion=torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [81, 122], gamma=0.1, last_epoch=-1)

    loss_log=[]
    acc_log=[]
    for epoch in range(20):
        loss_log.append(train(epoch,model,train_loader,optimizer,criterion,device))
        acc_log.append(test(epoch,model,valid_loader,device))
        scheduler.step()

    if acc_log[-1]>=max(acc_log):
        state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
        torch.save(state,'./checkpoints/FCNNMNIST.plt')
    np.savetxt('./log/loss_f.txt',loss_log,fmt='%.3f')
    np.savetxt('./log/acc_f.txt',acc_log,fmt='%.2f')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
