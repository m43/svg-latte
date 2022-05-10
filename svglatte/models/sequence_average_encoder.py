import torch
from torch import nn


class SequenceAverageEncoder(nn.Module):
    def __init__(self, feature_dim):
        super(SequenceAverageEncoder, self).__init__()
        self.output_size = self.feature_dim = feature_dim

    def forward(self, input_sequences, sequence_lengths=None):
        latte = torch.cat([seq[:length].mean(0).unsqueeze(0) for seq, length in zip(input_sequences, sequence_lengths)])
        return latte
