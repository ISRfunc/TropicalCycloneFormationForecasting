import torch 
import torch.nn as nn 

class DenseLayer3D(nn.Module): 

    def __init__(self, in_features, growth_rate: int, bn_size: int):
        super(DenseLayer3D, self).__init__()

        # 1x1x1 conv
        self.bn1 = nn.BatchNorm3d(num_features=in_features)
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv3d(in_channels=in_features, out_channels=growth_rate*bn_size, kernel_size=(1,1,1), stride=(1,1,1), bias=False)

        # 3x3x3 conv
        self.bn3 = nn.BatchNorm3d(num_features=growth_rate*4)
        self.conv3 = nn.Conv3d(in_channels=growth_rate*bn_size, out_channels=growth_rate, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.dropout = nn.Dropout(p=0.2)


    def forward(self, X): 
        out = self.conv1(self.gelu(self.bn1(X)))
        out = self.dropout(self.conv3(self.gelu(self.bn3(out))))

        return out

class DenseBlock3D(nn.Module):

    def __init__(self, n_layers, in_features, growth_rate: int=32, bn_size=4):
        super(DenseBlock3D, self).__init__()

        self.dense_layers = nn.ModuleList()
        
        for i in range(n_layers):
            self.dense_layers.append(DenseLayer3D(in_features=in_features + i*growth_rate, growth_rate=growth_rate, bn_size=bn_size))

    def forward(self, X):
        
        features = [X]

        for layer in self.dense_layers: 
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        
        return torch.cat(features, dim=1)

         

class TransitionBlock3D(nn.Module):

    def __init__(self, in_channels: int):
        super(TransitionBlock3D, self).__init__()

        self.bn = nn.BatchNorm3d(in_channels)
        self.gelu = nn.GELU()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=(1,1,1), stride=1, bias=False)
        self.pooling = nn.AvgPool3d(kernel_size=(2, 2,2), stride=(2, 2,2))

    def forward(self, X): 
        return self.pooling(self.conv1(self.gelu(self.bn(X))))

class LightDenseEncoder3D(nn.Module):

    def __init__(self, dummy_input: torch.Tensor, k: int=32):
        super(LightDenseEncoder3D, self).__init__()

        self.pre_dense_conv = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(7,7,7), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.pre_dense_pseudo_pooling = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(2,2,2), stride=(2,2,2), bias=False)

        self.dense_block = DenseBlock3D(n_layers=2, in_features=32, growth_rate=k, bn_size=4)

        self.transition_block = TransitionBlock3D(in_channels=32+2*k, out_channels=32+2*k)

        with torch.no_grad(): 
            dummy_output = self.transition_block(self.dense_block(self.pre_dense_pseudo_pooling(self.pre_dense_conv(dummy_input))))
            spatial_dim = dummy_output.shape[2:]
        
        self.global_pseudo_pooling = nn.Conv3d(in_channels=32+2*k, out_channels=32+2*k, kernel_size=spatial_dim, bias=False)


    def forward(self, X):

        feature_maps = self.pre_dense_pseudo_pooling(self.pre_dense_conv(X))
        feature_maps = self.dense_block(feature_maps)
        feature_maps = self.transition_block(feature_maps)
        feature_maps = self.global_pseudo_pooling(feature_maps)

        return torch.flatten(feature_maps, start_dim=1)

class DenseEncoder3D(nn.Module):

    def __init__(self, dummy_input: torch.Tensor, k: int=32):
        super(DenseEncoder3D, self).__init__()

        self.inital_features = 32

        self.pre_dense_conv = nn.Conv3d(in_channels=1, out_channels=self.inital_features, kernel_size=(7,7,7), stride=(2,2,2), padding=(1,1,1), bias=False)
        self.pre_dense_pooling = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2))

        self.dense_block_0 = DenseBlock3D(n_layers=6, in_features=self.inital_features, growth_rate=k, bn_size=4)

        self.transition_block = TransitionBlock3D(in_channels=self.inital_features+6*k)

        self.dense_block_1 = DenseBlock3D(n_layers=12, in_features=(self.inital_features+6*k)//2, growth_rate=k, bn_size=4)


        with torch.no_grad(): 
            dummy_output = self.dense_block_1(self.transition_block(self.dense_block_0(self.pre_dense_pooling(self.pre_dense_conv(dummy_input)))))
            spatial_dim = dummy_output.shape[2:]
        
        self.global_pooling = nn.AvgPool3d(kernel_size=spatial_dim)


    def forward(self, X):

        feature_maps = self.pre_dense_pooling(self.pre_dense_conv(X))
        feature_maps = self.dense_block_0(feature_maps)
        feature_maps = self.transition_block(feature_maps)
        feature_maps = self.dense_block_1(feature_maps)
        feature_maps = self.global_pooling(feature_maps)
        return torch.flatten(feature_maps, start_dim=1)



# Debug
if __name__ == "__main__": 
    dummy = torch.randn((5, 1, 42, 33, 33), dtype=torch.float) #batch, channels, iso, latitude, longitude


    # 3d variable encoder
    Encoder3D = DenseEncoder3D(k=12, dummy_input=torch.randn(dummy.shape))
    output = Encoder3D(dummy)
    print(output.shape)

    # print(Encoder3D)
    print(f"Number of params:{sum(p.numel() for p in Encoder3D.parameters())}")
    