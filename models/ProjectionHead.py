import torch 
import torch.nn as nn 


class ProjectionHead(nn.Module):
    def __init__(self,in_features: int, hidden_dim: int=32):
        super(ProjectionHead, self).__init__()

        self.linear_0 = nn.Linear(in_features=in_features, out_features=hidden_dim, bias=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(p=0.2)
        self.linear_1 = nn.Linear(in_features=hidden_dim, out_features=1, bias=True)

    
    def forward(self, X): 
        output = self.bn(self.gelu(self.dropout(self.linear_0(X))))
        output = self.linear_1(output)

        return output