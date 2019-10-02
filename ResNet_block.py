import torch
from torch import nn
from load_data import DataHelper
import torch.optim as optim
import time

class Residual(nn.Module):
    def __init__(self, input_channels, output_channels, use_1x1conv=False, strides=(1,1), **kwargs):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel_size=(1, 3), stride=strides, padding=(0, 1))
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=(1, 3), padding=(0, 1))
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(output_channels)
        self.bn2 = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        y = self.relu(y)
        return y

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
class Unsqueeze(nn.Module):
    def forward(self, x):
        return x.unsqueeze(1)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
  blk = []
  for i in range(num_residuals):
    if i == 0 and not first_block:
      blk.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=(1,2)))
    else:
      blk.append(Residual(num_channels, num_channels))
  return blk


b1 = nn.Sequential(
    Unsqueeze(),
    nn.Conv2d(1, 64, kernel_size=(1, 5), stride=(1, 3), padding=(0, 2)),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(1, 3), stride=(1,2), padding=(0,1))
)
b2=nn.Sequential(*resnet_block(64,64,2,first_block=True))

b3=nn.Sequential(*resnet_block(64,128,2))
b4=nn.Sequential(*resnet_block(128,128,2))

b5=nn.Sequential(*resnet_block(128,256,2))
b6=nn.Sequential(*resnet_block(256,256,2))

b7=nn.Sequential(*resnet_block(256,512,2))
b8=nn.Sequential(*resnet_block(512,512,2))

b9 = nn.Sequential(
    nn.AdaptiveAvgPool2d((8, 1)),
    Flatten(),
    nn.Linear(4096, 1),
    nn.Sigmoid()
)

net = nn.Sequential(
    b1,b2,b3,b4,b5,b6,b7,b8,b9
)

x = torch.randn((64, 8, 5000))
for layer in net:
    x = layer(x)
    print(layer.__class__.__name__, 'output shape: ', x.shape)


def init_weights(m):
    if type(m)==nn.Linear or type(m)==nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight)


helper = DataHelper()
train_iter, validation_iter = helper.get_train_and_validation_iter()
# device = torch.device('cuda:0')
device = torch.device('cpu')

net.to(device)
net.apply(init_weights)
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

torch.save(net.state_dict(), "net_resnet")
print(net.state_dict())