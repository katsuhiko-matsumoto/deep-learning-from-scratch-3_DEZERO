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
lr = 1.0

# データの読み込み
train_set = dezero.datasets.MNIST(train=True)
test_set = dezero.datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

#network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
        self.bn1 = L.BatchNorm()

    def forward(self, x):
        y = F.sigmoid(self.bn1(self.l1(x)))
        y = self.l2(y)
        return y


model = TwoLayerNet(hidden_size, 10)
optimizer = optimizers.SGD(lr).setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
    #if epoch % 10 == 0:
    #    print('loss:', loss.data)

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
