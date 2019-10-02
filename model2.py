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

class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 5000, 8)

net = torch.nn.Sequential(
    Reshape(),
    ## batch_size x 1 x 5000 x 8
    nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(9, 1), padding=(4, 1), stride=(10, 1)),
    ## batch_size x 4 x 500 x 10
    nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(5, 1), padding=(2, 1), stride=(4, 1)),
    ## batch_size x 16 x 125 x 12
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1), padding=(2, 1), stride=(4, 1)),
    ## batch_size x 32 x 64 x 14
    nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), padding=(2, 1), stride=(4,2)),
    ## batch_size 1 x 32 x 9 x 8
    Flatten(),
    ## batch_size x 2304
    nn.Linear(2304, 1200),
    nn.ReLU(),
    nn.Linear(1200, 800),
    nn.ReLU(),
    nn.Linear(800, 120),
    nn.ReLU(),
    nn.Linear(120, 1),
    nn.Sigmoid()
)

def init_weights(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)

device = torch.device('cuda:1')
net.apply(init_weights)
net.to(device)

def get_files():
    """
    :return:
    """
    train_path = Constant.TRAIN_DATA_PATH
    files = [f for f in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, f))]
    return files

helper = DataHelper()
files = helper.files
num_labels = helper.get_num_label_i(0) ## indicator patient has the 0th arrythmia or not
# for file, num_label_0 in zip(files, num_labels):
#     print(file+"\t"+str(num_label_0))



datas, labels = k_fold(5, files, num_labels)
validation_data = datas[0]
validation_label = labels[0]

train_data = []
train_labels = []
for i in range(1, 5):
    train_data += datas[i]
    train_labels += labels[i]

# print(train_labels)

# print(len(validation_data))
# print(len(train_data))
# print(len(validation_label))
# print(len(train_labels))
params = {
    'batch_size' : 64,
    'shuffle' : True,
    'num_workers' : 6
}


dataset = CardiogramDataset(train_data, train_labels)



train_iter = DataLoader(dataset, **params)
validation_iter = DataLoader(CardiogramDataset(validation_data, validation_label), **params)

max_epochs = 100
lr = 0.01
optimizer = optim.SGD(net.parameters(), lr=0.001)
criterion = nn.MSELoss()
train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
train_acc_sum = torch.tensor([0.0],dtype=torch.float32,device=device)
n = torch.tensor([0.0],dtype=torch.float32,device=device)
torch.save(net.state_dict(), "net0")

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


for epoch in range(1):
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


torch.save(net.state_dict(), "net")
print(net.state_dict())


# print(len(dataset))
























#
#
# helper = DataHelper()
# features, num_labels, labels = helper.get_patient_feature_and_label('2.txt')
# features = [[float(str) for str in item1 ] for item1 in features]
# features = torch.FloatTensor(features)
# features = features.to(device)
#
# for layer in net:
#     features = layer(features)
#     print(layer.__class__.__name__, 'output shape:', features.shape)
#
# print(features)

