import netCDF4
import numpy as np
import os

from dataset.defineLabel import *

from torch.utils.data import Dataset
import torch

from configs.configs_parser import load_config


class TropicalCycloneDataset(Dataset):
    def __init__(self, files, rawDataRoot, transforms = None, maxForecastTime = 36, fillMode = "outlier"):
        self.files = files
        self.dataRoot = rawDataRoot
        self.transforms = transforms

        # unit: hours
        self.maxForecastTime = maxForecastTime

        self.fillMode = fillMode

    def __getDataFromFile__(self, fileDir):

        f = netCDF4.Dataset(self.dataRoot + '/' + fileDir)
        if self.fillMode == "outlier":
            outlier_f = open("preprocessing/constants/outliers.txt", "r")
        elif self.fillMode == "mean":
            mean_f = load_config("preprocessing/constants/meansAndStds.yml") 
         
        usedVars = list(f.variables.keys())[:-4]

        varList = []

        for varName in usedVars:
            var = f.variables[varName][:]
            data = var.data
            mask = var.mask

            if self.fillMode == "outlier":
                fillVal = float(outlier_f.readline().split('\n')[0])
                data[mask == True] = fillVal
            elif self.fillMode == "zero":
                data[mask == True] = 0
            elif self.fillMode == "mean":
                data[mask == True] = float(mean_f[varName]['mean'])

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

        return varConcatTensor.unsqueeze(0), torch.as_tensor(label, dtype= torch.float)
