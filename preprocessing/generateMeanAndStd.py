import netCDF4
import numpy as np
import os
import random
from configs.configs_parser import load_config
from dataset.defineLabel import defineLabel

import yaml

varNames = None


def readDataFile(dataRoot, fileName):
    f = netCDF4.Dataset(dataRoot + '/' + fileName)

    usedVars = list(f.variables.keys())[:-4]

    global varNames
    if varNames == None:
        varNames = usedVars

    return f, usedVars


def __getMean__(varIdx, fileNames):

    samples = np.asarray([])

    random.shuffle(fileNames)
    sampledFileNames = fileNames[:]

    for fileName in sampledFileNames[:40]:

        f, usedVars = readDataFile(dataRoot, fileName)
            
        var = f.variables[usedVars[varIdx]][:]
        data = var.data
        mask = var.mask

        varData = data[mask == False].reshape(-1)

        samples = np.concatenate([samples, varData])

    return np.mean(samples)


def getMean(varIdx, fileNames, sampleIters = 10):
    
    samples = []

    for i in range(sampleIters):
        samples.append(__getMean__(varIdx, fileNames))

    samples = np.asarray(samples)

    return np.mean(samples)






def __getStd__(varIdx, fileNames):

    samples = np.asarray([])

    random.shuffle(fileNames)
    sampledFileNames = fileNames[:]

    for fileName in sampledFileNames[:40]:

        f, usedVars = readDataFile(dataRoot, fileName)
            
        var = f.variables[usedVars[varIdx]][:]
        data = var.data
        mask = var.mask

        varData = data[mask == False].reshape(-1)

        samples = np.concatenate([samples, varData])

    return np.std(samples)


def getStd(varIdx, fileNames, sampleIters = 10):
    
    samples = []

    for i in range(sampleIters):
        samples.append(__getStd__(varIdx, fileNames))

    samples = np.asarray(samples)

    return np.mean(samples)







if __name__ == "__main__":

    config = load_config("./configs/configs.yml")
    
    dataRoot = config['data']['root']
    maxForecastTime = config['data']['maxForecastTime']
    weighted_inputNorm = config['data']['weighted_inputNorm']

    w = os.walk(dataRoot)

    pos_files = []
    neg_files = []


    for (dirpath, dirnames, filenames) in w:

        for filename in filenames:

            pth = f"{os.path.relpath(dirpath, dataRoot)}/{filename}"
            category = defineLabel(pth, maxForecastTime)

            if category == 1:
                pos_files += [pth]
            else:
                neg_files += [pth]

    files = [pos_files] + [neg_files]

      
    varsInfo = {}
    f = open("preprocessing/constants/meansAndStds.yml", "w")

    for i in range(14):

        print(i)
        varInfo = {}

        if weighted_inputNorm:
            pos_mean = getMean(i, pos_files)
            neg_mean = getMean(i, neg_files)

            mean = (pos_mean + neg_mean) / 2.

            varInfo['mean'] = float(mean)


            pos_std = getStd(i, pos_files)
            neg_std = getStd(i, neg_files)

            std = (pos_std + neg_std) / 2.

            varInfo['std'] = float(std)

        else:
            mean = getMean(i, files)

            varInfo['mean'] = float(mean)


            std = getStd(i, files)

            varInfo['std'] = float(std)     


        varsInfo[varNames[i]] = varInfo
        

    yaml.dump(varsInfo, f, default_flow_style=False)

    f.close()

