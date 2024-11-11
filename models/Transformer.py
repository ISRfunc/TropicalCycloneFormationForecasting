import torch 
import torch.nn as nn
import math

class PositionalEncoder(nn.Module):
    def __init__(self, embedding_dim: int=128, max_seq_length: int=512, N: float=10000.0):
        super(PositionalEncoder, self).__init__()

        self.dropout = nn.Dropout(p=0.2)

        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2) * (-math.log(N) / embedding_dim))
        pe = torch.zeros(max_seq_length, embedding_dim)

        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        self.register_buffer('pe', pe)

    def forward(self, X): 
        
        X = X + self.pe[:X.size(1), :].unsqueeze(dim=0)

        return X

class AttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int=128, n_heads: int=4, dropout: bool=False, batch_first: bool=False):
        super(AttentionBlock, self).__init__()

        self.multiheaded_attention = nn.MultiheadAttention(embed_dim= embedding_dim, num_heads=n_heads, dropout=dropout, batch_first=batch_first)
        self.ln = nn.LayerNorm(normalized_shape=embedding_dim)



    def forward(self, X):

        attended_output, _ = self.multiheaded_attention(X, X, X)
        

        return self.ln(X + attended_output)

class MLP(nn.Module):

    def __init__(self, embedding_dim: int=128, projection_rate: int=4):
        super(MLP, self).__init__()

        self.up_projection = nn.Linear(in_features=embedding_dim, out_features=embedding_dim*projection_rate, bias=True)
        self.ln = nn.LayerNorm(normalized_shape=embedding_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.2)

        self.down_projection = nn.Linear(in_features=embedding_dim*projection_rate, out_features=embedding_dim, bias=True)



    def forward(self, X):
        output = self.gelu(self.dropout(self.up_projection(X)))
        output = self.ln(X + self.down_projection(output))

        return output

class CustomTransformer(nn.Module): 
    def __init__(self, embedding_dim: int=128, n_attention_heads: int=8, projection_rate: int=4):
        super(CustomTransformer, self).__init__()

        self.multiheaded_attention = AttentionBlock(embedding_dim=embedding_dim, n_heads=n_attention_heads, batch_first=True)

        self.mlp = MLP(embedding_dim=embedding_dim, projection_rate=projection_rate)

    def forward(self, X): 

        attended_output = self.multiheaded_attention(X)

        output = torch.stack([self.mlp(embedding) for embedding in attended_output])

        return output


# Debug
if __name__ == "__main__":
    dummy = torch.randn((5, 14, 196))

    embedding_dim = 196
    max_seq_length = 14

    pe = PositionalEncoder(embedding_dim=embedding_dim, max_seq_length=max_seq_length, N=10000)
    ct = CustomTransformer(embedding_dim=embedding_dim, n_attention_heads=14, projection_rate=4)

    print(pe(dummy).shape)
    print(ct(dummy).shape)
    # print(ct)
    # print(f"# Params: {sum(p.numel() for p in ct.parameters())}")