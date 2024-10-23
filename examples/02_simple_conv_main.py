import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import optimizers
from dezero import Model
import dezero.functions as F
import dezero.layers as L
from dezero import DataLoader

max_epoch = 100
batch_size = 30
hidden_size = 50
lr = 0.001

# データの読み込み transfomr=Noneでflattenせずに形状を維持する
train_set = dezero.datasets.MNIST(train=True, transform=None)
test_set = dezero.datasets.MNIST(train=False, transform=None)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

class SimbleConvNet(Model):
    def __init__(self, pretrained=False):
        super().__init__()
        self.conv1_1 = L.Conv2d(1, kernel_size=3, stride=1, pad=0)
        self.linear1 = L.Linear(100)
        self.linear2 = L.Linear(10)
    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.pooling(x, 2, 2)
        #print('shape:',x.shape)
        x = F.reshape(x, (x.shape[0], -1))
        #print('shape:',x.shape)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


model = SimbleConvNet()
optimizer = optimizers.SGD(lr).setup(model)

if dezero.cuda.gpu_enable:
    train_loader.to_gpu()
    test_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print('epoch: {}'.format(epoch+1))
    print('train loss: {}, accuracy: {}'.format(
        sum_loss / len(train_set), sum_acc / len(train_set)))

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print('test loss: {}, accuracy: {}'.format(
        sum_loss / len(test_set), sum_acc / len(test_set)))
