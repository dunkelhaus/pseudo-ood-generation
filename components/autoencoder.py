import torch
from torch import nn


class POGEncoder(nn.Module):
    """
    LSTM encoder.
    Adds noise too.
    """
    def __init__(self):
        super(POGEncoder, self).__init__()
        self.lstm = nn.LSTM(768, 100)

    def forward(self, x):
        z = self.lstm(x)
        epsilon = torch.randn(size=z.shape)
        return z + epsilon


class POGDecoder(torch.nn.Module):
    def __init__(self):
        super(POGDecoder, self).__init__()
        self.lstm = nn.LSTM(100, 768)

    def forward(self, z, x):
        pred_vec = nn.functional.log_softmax(self.lstm(z))
        return pred_vec, nn.functional.nll_loss(pred_vec, x)
