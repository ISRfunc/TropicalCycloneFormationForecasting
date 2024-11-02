from dataset.dataset import TropicalCycloneDataset
import os
from configs.configs_parser import load_config

if __name__ == "__main__":

    config = load_config("./configs/configs.yml")
    
    dataRoot = config['data']['root']
    maxForecastTime = config['data']['maxForecastTime']

    w = os.walk(dataRoot)

    files = []

    for (dirpath, dirnames, filenames) in w:

        files += [ f"{os.path.relpath(dirpath, dataRoot)}/{filename}" for filename in filenames]
      

    ds = TropicalCycloneDataset(files, dataRoot, maxForecastTime = maxForecastTime, fillOutlier = True)
    print(ds[0])
    print(len(ds))
