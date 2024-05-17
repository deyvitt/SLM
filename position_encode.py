# ok. Can you re-generate these codes differently so that it doesn't match the public code that includes:

#1. positional encoding
#2. encoder-decoder stack
#3. global and multihead attention with slidingwindow
#4. then the codes for outputing for the other model to pick up? (the aggregator model)?

#please break these up in 4 different files(modules) called:

#position_encode.py
#encoder-decoder.py
#attention.py
#bridge.py 
import torch
import torch.nn as nn

class Position_Encode(nn.Module):
    def __init__(self, sequence_length, embedding_size):
        super(Position_Encode, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        pos = torch.arange(sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * -(torch.log(torch.tensor(10000.0)) / embedding_size))
        self.pos_enc = torch.zeros(sequence_length, embedding_size)
        self.pos_enc[:, 0::2] = torch.sin(pos * div_term)
        self.pos_enc[:, 1::2] = torch.cos(pos * div_term)
        self.pos_enc = self.pos_enc.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        return x + self.pos_enc.to(x.device)

# Please tune the variables and values as to meet your requirements