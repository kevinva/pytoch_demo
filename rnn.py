import torch.nn as nn
import torch

class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim = 1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        # print('combined: ', combined.size())

        hidden = self.i2h(combined)
        # print('hidden: ', hidden.size())

        output = self.i2o(combined)
        # print('output1: ', output.size())

        output = self.softmax(output)
        # print('output2: ', output.size())

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)