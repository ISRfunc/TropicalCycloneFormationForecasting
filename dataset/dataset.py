import netCDF4
import numpy as np
import os

from dataset.defineLabel import *

from torch.utils.data import Dataset
import torch


class TropicalCycloneDataset(Dataset):
    def __init__(self, files, dataRoot, transform = None, maxForecastTime = 36, fillOutlier = False):
        self.files = files
        self.dataRoot = dataRoot
        self.transform = transform

        # unit: hours
        self.maxForecastTime = maxForecastTime

        self.fillOutlier = fillOutlier

    def __getDataFromFile__(self, fileDir):

        varDict = {}

        f = netCDF4.Dataset(self.dataRoot + '/' + fileDir)
        if self.fillOutlier:
            outlier_f = open("preprocessing/constants/outliers.txt", "r")
            
        usedVars = list(f.variables.keys())[:-4]

        for varName in usedVars:
            var = f.variables[varName][:]
            data = var.data
            mask = var.mask

            if self.fillOutlier:
                fillVal = float(outlier_f.readline().split('\n')[0])
                data[mask == True] = fillVal
            else:
                data[mask == True] = 0

            varData = data
            varDict[varName] = varData

        return varDict

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fileName = self.files[idx]

        label = defineLabel(fileName, self.maxForecastTime)

        input_vars = self.__getDataFromFile__(fileName) 

        transformed_input = self.transform(input_vars) if self.transform else input_vars

        return transformed_input, torch.as_tensor(label, dtype= torch.long)
