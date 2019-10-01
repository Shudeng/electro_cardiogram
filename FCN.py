import torch
from torch import nn as nn
from load_data import DataHelper

class MALSTM_FCN(nn.Module):
    def __init__(self):
        super(MALSTM_FCN, self).__init__()

    def forward(self, x):
        return x

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fcn = nn.Sequential(
            Reshape_For_FCN(), # change batch data to batch_size x channel x length shape

            ## first block
            nn.Conv1d(in_channels=8, out_channels=128, kernel_size=9, padding=4),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            squeeze_excite_block(inch=128, r=4),

            # second block
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            squeeze_excite_block(inch=256, r=8),

            ## third block
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),

            ## global pooling
            GlobalAvgPool()
        )

    def forward(self, x):
        return self.fcn(x)

class Reshape_For_FCN(torch.nn.Module):
    def __init__(self):
        super(Reshape_For_FCN, self).__init__()
    def forward(self, x):
        return x.view(-1, 8, 5000)

class GlobalAvgPool(nn.Module):
    def __init__(self):
        super(GlobalAvgPool, self).__init__()
    def forward(self, x):
        # *(a) transfor list a to parameters of function
        # mean(i) calculate the means by i-th dimension
        # return x.view(*(x.shape[:-1]), -1).mean(-1)
        return x.mean(-1)

class squeeze_excite_block(nn.Module):
    def __init__(self, inch, r):
        super(squeeze_excite_block, self).__init__()
        self.se = nn.Sequential(
            GlobalAvgPool(), ## batch_size x channels. Here channels is also inch
            nn.Linear(inch, inch // r), ## batch_size x inch//r
            nn.ReLU(inplace=True), ## batch_size x inch//r
            nn.Linear(inch // r, inch), ## batch_size x inch
            nn.Sigmoid() ## batch_size x inch
        )

    def forward(self, x):
        # self.se(x) is in batch_size x inch
        # se_weight is in batch_size x inch x 1
        se_weight = self.se(x).unsqueeze(-1)
        return x.mul(se_weight)

class LSTM_RESHAPE(nn.Module):
    def forward(self, x):
        return x.view((-1, x.shape[2], x.shape[1]))

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.Sequential(
            LSTM_RESHAPE(),

        )

    def forward(self, x):
        return x

helper = DataHelper()
train_iter, test_iter = helper.get_train_and_validation_iter()
for X, y in train_iter:
    x = X
    break
# print(x.shape)
# lstm_reshape = LSTM_RESHAPE()
# x = lstm_reshape(x)
# print(x.shape)
#
#
# fcn = FCN()
# x = fcn(x)
# print("x.shape: {}".format(x.shape))





# fcn = nn.Sequential(
#     Reshape_For_FCN(), # change batch data to batch_size x channel x length shape
#
#     ## first block
#     nn.Conv1d(in_channels=8, out_channels=128, kernel_size=9, padding=4),
#     nn.BatchNorm1d(num_features=128),
#     nn.ReLU(),
#     squeeze_excite_block(inch=128, r=4),
#
#     # second block
#     nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2),
#     nn.BatchNorm1d(num_features=256),
#     nn.ReLU(),
#     squeeze_excite_block(inch=256, r=8),
#
#     ## third block
#     nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
#     nn.BatchNorm1d(num_features=128),
#     nn.ReLU(),
#     GlobalAvgPool()
# )

# for layer in fcn:
#     x = layer(x)
#     print("layer name:{}, \t {}".format(layer.__class__.__name__, x.shape))
