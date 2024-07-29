import torch
from torch import nn

class EBD(nn.Module):
    def __init__(self):
        super(EBD, self).__init__()
        self.word_ebd = nn.Embedding(28, 24)
        self.pos_ebd = nn.Embedding(12, 24)
        self.pos_t = torch.arange(0, 12).reshape(1, 12)

    def forward(self, X:torch.tensor):
        return self.word_ebd(X) + self.pos_ebd(self.pos_t)

if __name__ == "__main__" :
    a = torch.ones((2,12)).long()
    ebd = EBD()
    b = ebd(a)
    print(b.shape)

    pass