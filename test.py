import torch
from torch import nn
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

net = squeeze_excite_block(4, 2)
x = [
    [
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4]
    ],
    [
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4]
    ]
]

x = torch.tensor(x, dtype=torch.float32)
print(x.shape)

se = squeeze_excite_block(4, 2)
y = se(x)
print(y.shape)
print(y)