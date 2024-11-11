import netCDF4
import numpy as np
import os
import random
from configs.configs_parser import load_config


def __getFillValue__(varIdx, fileNames):

    samples = np.asarray([])

    random.shuffle(fileNames)
    sampledFileNames = fileNames[:20]

    for fileName in sampledFileNames:

        f = netCDF4.Dataset(dataRoot + '/' + fileName)
        usedVars = list(f.variables.keys())[:-4]
            
        var = f.variables[usedVars[varIdx]][:]
        data = var.data
        mask = var.mask

        varData = data[mask == False].reshape(-1)

        samples = np.concatenate([samples, varData])

    hist, bin_edges = np.histogram(samples, bins = 30)

    min_hist = np.argmin(hist)

    return (bin_edges[min_hist] + bin_edges[min_hist+1]) / 2


def getFillValue(varIdx, fileNames, sampleIters = 10, binWidth = 3):
    
    samples = []

    for i in range(sampleIters):

        samples.append(__getFillValue__(varIdx, fileNames))

    samples = np.asarray(samples)
    ret = samples[0]

    hist, bin_edges = np.histogram(samples, bins = (sampleIters // binWidth))

    hist = hist.astype(np.float64)
    hist[hist == 0] = np.inf

    min_hist = np.argmin(hist)

    for sample in samples:
        if sample > bin_edges[min_hist] and sample < bin_edges[min_hist+1]:
            ret = sample

    return ret        


if __name__ == "__main__":

    config = load_config("./configs/dataset_configs.yml")
    
    dataRoot = config['data']['rootRawData']

    w = os.walk(dataRoot)

    files = []


    for (dirpath, dirnames, filenames) in w:

        files += [ f"{os.path.relpath(dirpath, dataRoot)}/{filename}" for filename in filenames]


    f = open("preprocessing/constants/outliers.txt", "w")

    for i in range(14):
        
        fillVal = getFillValue(i, files).astype('|S10').decode('UTF-8') + '\n'
        
        f.write(fillVal)

    f.close()

