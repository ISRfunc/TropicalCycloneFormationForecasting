from .Encoder2D import DenseEncoder2D, LightDenseEncoder2D
from .Encoder3D import DenseEncoder3D, LightDenseEncoder3D
from .Transformer import CustomTransformer, PositionalEncoder
from .ProjectionHead import ProjectionHead
from .constants import variable_names, range_list

import torch 
import torch.nn as nn



# Shared encoder
class Arch1(nn.Module):
    def __init__(self, dummy_input3d: torch.Tensor, dummy_input2d: torch.Tensor):
        super(Arch1, self).__init__()

        self.k = 12
        self.encoder3d = DenseEncoder3D(dummy_input=dummy_input3d, k=self.k)
        self.encoder2d = DenseEncoder2D(dummy_input=dummy_input2d, k=self.k)

    def forward(self, X): 
        
        #PHIS, PS, SLP is 2d variable, the rest is 3d
        #var_name = ["EPV", "H", "O3", "OMEGA", "PHIS", "PS", "QI", ]
        E_EPV    = self.encoder3d(X["EPV"])
        E_H      = self.encoder3d(X["H"])
        E_O3     = self.encoder3d(X["O3"])
        E_OMEGA  = self.encoder3d(X["OMEGA"])
        E_PHIS   = self.encoder2d(X["PHIS"])
        E_PS     = self.encoder2d(X["PS"])
        E_QI     = self.encoder3d(X["QI"])
        E_QL     = self.encoder3d(X["QL"])
        E_QV     = self.encoder3d(X["QV"])
        E_RH     = self.encoder3d(X["RH"])
        E_SLP    = self.encoder2d(X["SLP"])
        E_T      = self.encoder3d(X["T"])
        E_U      = self.encoder3d(X["U"])
        E_V      = self.encoder3d(X["V"])

        return torch.stack([E_EPV, E_H, E_O3, E_OMEGA, E_PHIS, E_PS, E_QI, E_QL, E_QV, E_RH, E_SLP, E_T, E_U, E_V], dim=1)
    

# Independent encoders
class Arch2(nn.Module):
    def __init__(self, dummy_input3d: torch.Tensor, dummy_input2d: torch.Tensor):
        super(Arch2, self).__init__()

        self.encoder3d_EPV = DenseEncoder3D(dummy_input=dummy_input3d, k=16)
        self.encoder3d_H = DenseEncoder3D(dummy_input=dummy_input3d, k=16)
        self.encoder3d_O3 = DenseEncoder3D(dummy_input=dummy_input3d, k=16)
        self.encoder3d_OMEGA = DenseEncoder3D(dummy_input=dummy_input3d, k=16)
        self.encoder3d_QI = DenseEncoder3D(dummy_input=dummy_input3d, k=16)
        self.encoder3d_QL = DenseEncoder3D(dummy_input=dummy_input3d, k=16)
        self.encoder3d_QV = DenseEncoder3D(dummy_input=dummy_input3d, k=16)
        self.encoder3d_RH = DenseEncoder3D(dummy_input=dummy_input3d, k=16)
        self.encoder3d_T = DenseEncoder3D(dummy_input=dummy_input3d, k=16)
        self.encoder3d_U = DenseEncoder3D(dummy_input=dummy_input3d, k=16)
        self.encoder3d_V = DenseEncoder3D(dummy_input=dummy_input3d, k=16)


        self.encoder2d_PHIS = DenseEncoder2D(dummy_input=dummy_input2d, k=16)
        self.encoder2d_PS = DenseEncoder2D(dummy_input=dummy_input2d, k=16)
        self.encoder2d_SLP = DenseEncoder2D(dummy_input=dummy_input2d, k=16)


    def forward(self, X): 
        #PHIS, PS, SLP is 2d variable, the rest is 3d
        #var_name = ["EPV", "H", "O3", "OMEGA", "PHIS", "PS", "QI", ]
        E_EPV    = self.encoder3d_EPV(X["EPV"])
        E_H      = self.encoder3d_H(X["H"])
        E_O3     = self.encoder3d_O3(X["O3"])
        E_OMEGA  = self.encoder3d_OMEGA(X["OMEGA"])
        E_PHIS   = self.encoder2d_PHIS(X["PHIS"])
        E_PS     = self.encoder2d_PS(X["PS"])
        E_QI     = self.encoder3d_QI(X["QI"])
        E_QL     = self.encoder3d_QL(X["QL"])
        E_QV     = self.encoder3d_QV(X["QV"])
        E_RH     = self.encoder3d_RH(X["RH"])
        E_SLP    = self.encoder2d_SLP(X["SLP"])
        E_T      = self.encoder3d_T(X["T"])
        E_U      = self.encoder3d_U(X["U"])
        E_V      = self.encoder3d_V(X["V"])

        return torch.stack([E_EPV, E_H, E_O3, E_OMEGA, E_PHIS, E_PS, E_QI, E_QL, E_QV, E_RH, E_SLP, E_T, E_U, E_V], dim=1)
    

class FullModel(nn.Module):
    def __init__(self, arch: str):
        super(FullModel, self).__init__()

        dummy_input3d = torch.randn(1, 1, 42, 33, 33)
        dummy_input2d = torch.randn(1, 1, 33, 33)

        self.embedding_dim = 196 
        self.max_seq_length = 14
        self.n_attention_heads = 28
        self.var_names = variable_names
        self.range_list = range_list

        if arch == "arch1": 
            self.arch = Arch1(dummy_input2d=dummy_input2d, dummy_input3d=dummy_input3d)
        else: 
            self.arch = Arch2(dummy_input2d=dummy_input2d, dummy_input3d=dummy_input3d)

        self.positional_encoder = PositionalEncoder(embedding_dim=self.embedding_dim, max_seq_length=self.max_seq_length)
        self.transformer_0 = CustomTransformer(embedding_dim=self.embedding_dim, n_attention_heads=self.n_attention_heads, projection_rate=4)
        self.transformer_1 = CustomTransformer(embedding_dim=self.embedding_dim, n_attention_heads=self.n_attention_heads, projection_rate=4)

        self.ffn = ProjectionHead(in_features=self.embedding_dim*self.max_seq_length, hidden_dim=32)

    def unconcat(self, X):
        unconcat_X = {}
        for (var_name, ran) in zip(self.var_names, self.range_list):
            unconcat_X[var_name] = X[:, :, ran[0]:ran[1], :, :] if (ran[1] - ran[0] != 1) else X[:, :, ran[0]:ran[1], :, :].squeeze(dim=2)
        return unconcat_X
    
    def forward(self, X):
        unconcat_X = self.unconcat(X)
        features = self.arch(unconcat_X)
        
        attended_features = torch.flatten(self.transformer_1(self.transformer_0(self.positional_encoder(features))), start_dim=1)
        logits = self.ffn(attended_features)
        return logits
        


if __name__ == "__main__":

    dummy_batch = 32
    example_input = {
        "EPV":   torch.randn(dummy_batch, 1, 42, 33, 33),
        "H":     torch.randn(dummy_batch, 1, 42, 33, 33),
        "O3":    torch.randn(dummy_batch, 1, 42, 33, 33),
        "OMEGA": torch.randn(dummy_batch, 1, 42, 33, 33),
        "PHIS":  torch.randn(dummy_batch, 1, 1, 33, 33),
        "PS":    torch.randn(dummy_batch, 1, 1, 33, 33),
        "QI":    torch.randn(dummy_batch, 1, 42, 33, 33),
        "QL":    torch.randn(dummy_batch, 1, 42, 33, 33),
        "QV":    torch.randn(dummy_batch, 1, 42, 33, 33),
        "RH":    torch.randn(dummy_batch, 1, 42, 33, 33),
        "SLP":   torch.randn(dummy_batch, 1, 1, 33, 33),
        "T":     torch.randn(dummy_batch, 1, 42, 33, 33),
        "U":     torch.randn(dummy_batch, 1, 42, 33, 33),
        "V":     torch.randn(dummy_batch, 1, 42, 33, 33) 
    }

    example_input1 = torch.cat([var for var in example_input.values()], dim=2)  # torch dataloader return this 

    a1 = FullModel(arch="arch1")
    print(a1(example_input1).shape)
    print(f"Number of params:{sum(p.numel() for p in a1.parameters())}")


