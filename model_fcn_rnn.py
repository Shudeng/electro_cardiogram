import torch
from torch import nn as nn
from load_data import DataHelper
from cardiogram_dataset import k_fold
import os
import Constant
from load_data import DataHelper
from cardiogram_dataset import CardiogramDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from FCN import FCN
from Seq2seq import Encoder
from Seq2seq import RESHAPE


class FCN_RNN(nn.Module):
    def __init__(self):
        super(FCN_RNN, self).__init__()
        self.fcn = FCN()
        self.reshape = RESHAPE()
        self.encoder1 = Encoder(5000, 512)
        self.encoder2 = Encoder(512, 32)
        self.linear = nn.Linear(384, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        # fcn
        x_fcn = self.fcn(x) # size of x_fcn is batch_size x 128

        # rnn
        x_reshape = self.reshape(x)
        outputs, hidden = self.encoder1(x_reshape)
        outputs, hidden = self.encoder2(outputs) # outputs size is batch_size x 8 x 32
        # change size of outputs to batch_size x 256
        outputs = outputs.reshape(-1, 256)

        # concat fcn and rnn
        x_cat = torch.cat((x_fcn, outputs), dim=1)

        ## linear + softmax
        x_cat = self.linear(x_cat)
        x_cat = self.sigmoid(x_cat)
        return x_cat


def init_weights(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)


helper = DataHelper()
train_iter, validation_iter = helper.get_train_and_validation_iter()
for X, y in train_iter:
    x = X
    break
# net = Encoder(5000, 512)
net = FCN_RNN()
# device = torch.device('cuda:0')
device = torch.device('cpu')

net.to(device)
x = x.to(device)
net.apply(init_weights)

x =  net(x)
# print(net)
# for param in net.parameters():
#     print(type(param.data), param.size())


# print(x.shape)
# print(x)
# exit(0)

params = {
    'batch_size' : 64,
    'shuffle' : True,
    'num_workers' : 6
}

max_epochs = 100
lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=0.001)
criterion = nn.MSELoss()
train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
train_acc_sum = torch.tensor([0.0],dtype=torch.float32,device=device)
n = torch.tensor([0.0],dtype=torch.float32,device=device)


def evaluate_accuracy(data_iter, net, device):
    acc_sum, n = torch.tensor([0], dtype=torch.float32, device=device), 0
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        net.eval()
        with torch.no_grad():
            y = y.long()
            y_hat = net(X)
            acc_sum += torch.sum(
                (
                        (y_hat>0.5).long()== y.long()
                ).long()
            )
            n+=y.shape[0]
    return acc_sum.item()/n


for epoch in range(100):
    start = time.time()
    for X, labels in train_iter:
        # print(X.shape, labels.shape)
        net.train()
        optimizer.zero_grad()
        X, y = X.to(device), labels.to(device)
        # print(y.shape[0])
        y_hat = net(X)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y = y.long()
            train_l_sum += loss.float()
            train_acc_sum += torch.sum(
                (
                        (y_hat>0.5).long()
                        == y.long()
                ).long()
            )
            n += y.shape[0]
    validation_acc = evaluate_accuracy(validation_iter, net, device)
    print("epoch {}, loss: {}, \t train acc: {} \t test acc: {} \t time: {}".format(
        epoch + 1, train_l_sum / n, train_acc_sum / n, validation_acc, time.time() - start
    ))

torch.save(net.state_dict(), "net_fcn_rnn")
print(net.state_dict())