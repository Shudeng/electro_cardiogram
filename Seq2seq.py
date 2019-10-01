import torch
from torch import nn
from load_data import DataHelper
class RESHAPE(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size,
                 n_layers=2, dropout =0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, n_layers,
                          dropout=dropout , batch_first=True, bidirectional=True)
    def forward(self, x, hidden=None):
        outputs, hidden = self.gru(x, hidden)
        # sum bidirectional outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden

class Attention(nn.Module):

    def forward(self, x):
        return x

class Decoder(nn.Module):

    def forward(self, x):
        return x

class Seq2seq(nn.Module):

    def forward(self, x):
        return x

# helper = DataHelper()
# train_iter, test_iter = helper.get_train_and_validation_iter()
# for X, y in train_iter:
#     x = X
#     break
#
# print(x.shape)
# reshape = RESHAPE()
# x = reshape(x)
# print("after reshape: ",x.shape)
#
# encoder = Encoder(5000, 512)
# outputs, hidden = encoder(x)
#
# print("after encoder:", outputs.shape)
# print("after encoder:", hidden.shape)
#
# encoder = Encoder(512, 32)
# outputs, hidden = encoder(outputs)
#
# print("after encoder1:", outputs.shape)
# print("after encoder1:", hidden.shape)