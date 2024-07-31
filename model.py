import torch
from torch import nn

class EBD(nn.Module):
    def __init__(self):
        super(EBD, self).__init__()
        self.word_ebd = nn.Embedding(29, 24)
        self.pos_ebd = nn.Embedding(12, 24)
        self.pos_t = torch.arange(0, 12).reshape(1, 12)
        #将词编码后转换为词向量，然后再对位置进行编码，这里采用简单的方法来编码

    def forward(self, X:torch.tensor):
        return self.word_ebd(X) + self.pos_ebd(self.pos_t[:, :X.shape[-1]])#长度不一定是12
        #更改了词向量的位置，让他保存了前面的信息

def attention(Q, K, V, M:torch.tensor):
    A = Q @ K.transpose(-1, -2) / (Q.shape[-1] ** 0.5)
    M = M.unsqueeze(1)
    A.masked_fill(M == 0, -torch.tensor(float('inf')))
    A = torch.softmax(A, dim = -1)
    O = A @ V
    return O
    #注意力相乘

def transpose_qkv(QKV):
    QKV = QKV.reshape(QKV.shape[0], QKV.shape[1], 4, 6)
    QKV = QKV.transpose(-2, -3)
    return QKV
    #多头注意力机制

def transpost_o(O):
    O = O.transpose(-2, -3)
    O = O.reshape(O.shape[0], O.shape[1], 24)
    return O
    #转换回原来的维度

class Attention_block(nn.Module):
    def __init__(self):
        super(Attention_block, self).__init__()
        self.Wq = nn.Linear(24, 24, bias=False)
        self.Wk = nn.Linear(24, 24, bias=False)
        self.Wv = nn.Linear(24, 24, bias=False)
        self.Wo = nn.Linear(24, 24, bias=False)
        #里面的qkv均为可学习参数

    def forward(self, X, M:torch.tensor):
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)
        Q, K, V = transpose_qkv(Q), transpose_qkv(K), transpose_qkv(V)
        O = attention(Q, K, V, M)
        O = transpost_o(O)
        O = self.Wo(O)
        return O

class AddNorm(nn.Module):
    def __init__(self):
        super(AddNorm, self).__init__()
        self.add_norm = nn.LayerNorm(24)
        self.dropout = nn.Dropout(0.1)
        #防止过拟合

    def forward(self, X, X1):
        X += X1
        X = self.add_norm(X)
        X = self.dropout(X)
        return X
        #类比了残差网络，同时还归一化

class Pos_FNN(nn.Module):
    def __init__(self):
        super(Pos_FNN, self).__init__()
        self.lin_1 = nn.Linear(24, 48, bias=False)
        self.relu_1 = nn.ReLU()
        self.lin_2 = nn.Linear(48, 24, bias=False)
        self.relu_2 = nn.ReLU()

    def forward(self, X):
        X = self.lin_1(X)
        X = self.relu_1(X)
        X = self.lin_2(X)
        X = self.relu_2(X)
        return X
        #实现前馈网络

class Encoder_block(nn.Module):
    def __init__(self):
        super(Encoder_block, self).__init__()
        self.attention = Attention_block()
        self.add_norm_1 = AddNorm()
        self.FNN = Pos_FNN()
        self.add_norm_2 = AddNorm()

    def forward(self, X, I_m):
        I_m = I_m.unsqueeze(-2)
        X_1 = self.attention(X, I_m)
        X = self.add_norm_1(X, X_1)
        X_1 = self.FNN(X)
        X = self.add_norm_2(X, X_1)
        return X

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.ebd = EBD()
        self.encoder_blks = nn.Sequential()
        self.encoder_blks.append(Encoder_block())
        self.encoder_blks.append(Encoder_block())

    def forward(self, X, I_m):
        X = self.ebd(X)
        for encoder_blk in self.encoder_blks:
            X = encoder_blk(X, I_m)
        return X

class CrossAttention_block(nn.Module):
    def __init__(self):
        super(CrossAttention_block, self).__init__()
        self.Wq = nn.Linear(24, 24, bias=False)
        self.Wk = nn.Linear(24, 24, bias=False)
        self.Wv = nn.Linear(24, 24, bias=False)
        self.Wo = nn.Linear(24, 24, bias=False)
        #里面的qkv均为可学习参数

    def forward(self, X, X_en, I_m):
        Q, K, V = self.Wq(X), self.Wk(X_en), self.Wv(X_en)
        Q, K, V = transpose_qkv(Q), transpose_qkv(K), transpose_qkv(V)
        O = attention(Q, K, V, I_m)
        O = transpost_o(O)
        O = self.Wo(O)
        return O

class Decoder_blk(nn.Module):
    def __init__(self):
        super(Decoder_blk, self).__init__()
        self.attention = Attention_block()#跟encoder里面的注意力一样
        self.add_norm_1 = AddNorm()
        self.cross_attention = CrossAttention_block()#这里的多头注意力跟attention_block不一样，接收的不止有ebd之后的X，还有编码器编码后的X_en
        self.add_norm_2 = AddNorm()
        self.FNN = Pos_FNN()
        self.add_norm_3 = AddNorm()
        mask_matrix = torch.ones(12, 12)
        self.tril_mask = torch.tril(mask_matrix).unsqueeze(0)

    def forward(self, X_t, O_m, X_en, I_m):
        O_m = O_m.unsqueeze(-2)
        I_m = I_m.unsqueeze(-2)
        X_1 = self.attention(X_t, O_m * self.tril_mask[:, :O_m.shape[-1], :O_m.shpe[-1]])
        X_t = self.add_norm_1(X_t, X_1)
        X_1 = self.cross_attention(X_t, X_en, I_m)
        X_t = self.add_norm_2(X_t, X_1)
        X_1 = self.FNN(X_t)
        X_t = self.add_norm_3(X_t, X_1)
        return X_t

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.ebd = EBD()#先进行编码
        self.decoder_blks = nn.Sequential()
        self.decoder_blks.append(Decoder_blk())
        self.decoder_blks.append(Decoder_blk())
        self.dense = nn.Linear(24, 28, bias=False)#将他能够映射到28个字符里面

    def forward(self, X_t, O_m, X_en, I_m):
        X_t = self.ebd(X_t)
        for layer in self.decoder_blks:
            X = layer(X_t, O_m, X_en, I_m)
        X_t = self.dense(X_t)
        return X_t

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, X_s, I_m, X_t, O_m):
        X_en = self.encoder(X_s, I_m)
        X = self.decoder(X_t, O_m, X_en, I_m)
        return X

if __name__ == "__main__" :
    a = torch.ones((2, 12)).long()
    b = torch.ones((2, 1)).long()
    model = Transformer()
    o = model(a, b)
    pass