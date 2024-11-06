from torchvision import transforms as T
from PIL import Image
from configs.configs_parser import load_config

from torchvision.transforms.v2 import functional as F, Transform
from torch.nn.functional import affine_grid, grid_sample

import numpy as np
import random 
import torch



def getVarMeanAndStd():

    config = load_config("preprocessing/constants/meansAndStds.yml")    

    mean = []
    std = []
    isoChannels = []

    for var in config:
        mean += [config[var]['mean']]
        std += [config[var]['std']]
        isoChannels += [config[var]['isoChannels']]

    return mean, std, isoChannels

def getNormTrans(varMean, varStd, varIsoChannels):

    norm_Transformers = []

    for m, std, isoChannels in zip(varMean, varStd, varIsoChannels):

        norm_Transformer = T.Compose([
            T.Normalize(mean=[m] * isoChannels, std=[std] * isoChannels)
        ])

        norm_Transformers.append(norm_Transformer)

    return norm_Transformers



class AddGaussianNoise(object):
   
    def __init__(self, mean, sigma, clip=True, clipRange=1., inplace = False):
        self.mean = mean
        self.sigma = sigma
        self.clip = clip
        self.clipRange = clipRange
        self.inplace = inplace

    def __call__(self, tensor):

        if not self.inplace:
            tensor = tensor.clone()

        dtype = tensor.dtype
        noise = self.mean + torch.randn_like(tensor, dtype = dtype, device = tensor.device) * self.sigma
        out = tensor + noise
        if self.clip:
            out = torch.clamp(out, -self.clipRange / 2., self.clipRange / 2.)

        return out

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.sigma)



class RandomShear(object):
   
    def __init__(self, randomRange, inplace = False):
        self.randomRange = randomRange
        self.inplace = inplace

    def __call__(self, tensor):

        if not self.inplace:
            tensor = tensor.clone()

        tensor = tensor.unsqueeze(0)
        dtype = tensor.dtype
        
        rot = torch.rand((2,2), dtype=dtype, device = tensor.device) * self.randomRange + (1. - self.randomRange / 2.)
        trans = torch.zeros((2, 1), dtype=dtype, device = tensor.device)
        affine_matrix = torch.cat((rot, trans), 1)
       
        grid = affine_grid(affine_matrix.unsqueeze(0), tensor.size(), align_corners=False)

        output = grid_sample(tensor, grid, align_corners=False)

        return torch.squeeze(output, 0)

    def __repr__(self):
        return self.__class__.__name__ + '(randomRange={0})'.format(self.randomRange)



class RandomTranslate(object):
   
    def __init__(self, randomRange, inplace = False):
        self.randomRange = randomRange
        self.inplace = inplace

    def __call__(self, tensor):

        if not self.inplace:
            tensor = tensor.clone()

        tensor = tensor.unsqueeze(0)
        dtype = tensor.dtype
        
        rot = torch.tensor([[1,0], [0,1]], dtype=dtype, device = tensor.device)
        trans = torch.rand((2,1), dtype=dtype, device = tensor.device) * self.randomRange + (0. - self.randomRange / 2.)
        affine_matrix = torch.cat((rot, trans), 1)
       
        grid = affine_grid(affine_matrix.unsqueeze(0), tensor.size(), align_corners=False)

        output = grid_sample(tensor, grid, align_corners=False)

        return torch.squeeze(output, 0)

    def __repr__(self):
        return self.__class__.__name__ + '(randomRange={0})'.format(self.randomRange)



class RandomRotate(object):
   
    def __init__(self, randomRange, inplace = False):
        self.randomRange = randomRange
        self.inplace = inplace

    def __call__(self, tensor):

        if not self.inplace:
            tensor = tensor.clone()

        tensor = tensor.unsqueeze(0)
        dtype = tensor.dtype

        theta = random.random() * self.randomRange + (0. - self.randomRange / 2.)
        
        affine_matrix = torch.tensor([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0]], dtype=dtype, device = tensor.device)
       
        grid = affine_grid(affine_matrix.unsqueeze(0), tensor.size(), align_corners=False)

        output = grid_sample(tensor, grid, align_corners=False)

        return torch.squeeze(output, 0)

    def __repr__(self):
        return self.__class__.__name__ + '(randomRange={0})'.format(self.randomRange)


class RandomSpatialTrans(RandomShear, RandomTranslate, RandomRotate):

    def __init__(self, randomRange, inplace = False):

        self.select = random.randint(0, 2)

        if self.select == 0:
            self.chosenClass = RandomShear
        elif self.select == 1:
            self.chosenClass = RandomTranslate
        elif self.select == 2:
            self.chosenClass = RandomRotate

        self.chosenClass.__init__(self, randomRange, inplace)

    def __call__(self, tensor):
        
        return self.chosenClass.__call__(self, tensor)

    def __repr__(self):
        return RandomSpatialTrans.__bases__[self.select].__name__ + '(randomRange={0})'.format(self.randomRange)


def getTrainAugmenter(norm_Transformers):

    augmenters = []

    for norm_Transformer in norm_Transformers:
        
        augmenter = T.Compose([
            norm_Transformer,
            AddGaussianNoise(0., 0.1, clip = True, clipRange=0.1),
            RandomSpatialTrans(0.3)
        ])

        augmenters.append(augmenter)
    

    return augmenters

def getTestAugmenter(norm_Transformers):

    augmenters = []

    for norm_Transformer in norm_Transformers:

        augmenter = T.Compose([
            norm_Transformer
        ])

        augmenters.append(augmenter)

    return augmenters

