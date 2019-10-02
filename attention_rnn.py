import torch
from torch import nn
device = torch.device('cpu')
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, inputs, hidden=None):
        output, hidden = self.gru(inputs, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

x = torch.randn((64, 5000, 8))
encoder = Encoder(8, 10)
print(encoder)

output, hidden = encoder(x)
print(output.shape)
print(hidden.shape)