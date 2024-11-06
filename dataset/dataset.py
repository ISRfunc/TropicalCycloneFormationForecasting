import netCDF4
import numpy as np
import os

from dataset.defineLabel import *

from torch.utils.data import Dataset
import torch


class TropicalCycloneDataset(Dataset):
    def __init__(self, files, rawDataRoot, transforms = None, maxForecastTime = 36, fillOutlier = False):
        self.files = files
        self.dataRoot = rawDataRoot
        self.transforms = transforms

        # unit: hours
        self.maxForecastTime = maxForecastTime

        self.fillOutlier = fillOutlier

    def __getDataFromFile__(self, fileDir):

        f = netCDF4.Dataset(self.dataRoot + '/' + fileDir)
        if self.fillOutlier:
            outlier_f = open("preprocessing/constants/outliers.txt", "r")
            
        usedVars = list(f.variables.keys())[:-4]

        varList = []

        for varName in usedVars:
            var = f.variables[varName][:]
            data = var.data
            mask = var.mask

            if self.fillOutlier:
                fillVal = float(outlier_f.readline().split('\n')[0])
                data[mask == True] = fillVal
            else:
                data[mask == True] = 0

            varData = torch.Tensor(data)
            if len(varData.size()) == 2:
                varData = varData.unsqueeze(0)
                
            varList.append(varData)

        return varList

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fileName = self.files[idx]

        label = defineLabel(fileName, self.maxForecastTime)

        input_vars = self.__getDataFromFile__(fileName) 

        if self.transforms:
            for i in range(len(input_vars)):
                input_vars[i] = self.transforms[i](input_vars[i])    

        varConcatTensor = torch.cat(input_vars, 0)

        return varConcatTensor, torch.as_tensor(label, dtype= torch.long)
