from dataset.dataset import TropicalCycloneDataset
from dataset.transform import *
from configs.configs_parser import load_config

from torch.utils.data import DataLoader


def read_data_list(data_path, file_name):
    with open(f"{data_path}/{file_name}", "r", encoding="utf-8") as f: 
        data_list = f.read().splitlines()
    
    return data_list


if __name__ == "__main__":

    config = load_config("./configs/configs.yml")
    
    rootRawData = config['data']['rootRawData']
    rootSplitData = config['data']['rootSplitData']
    maxForecastTime = config['data']['maxForecastTime']

    trainSet = read_data_list(rootSplitData, "train.txt")
    valSet = read_data_list(rootSplitData, "val.txt")
    testSet = read_data_list(rootSplitData, "test.txt")


    
    varMean, varStd, varIsoChannels = getVarMeanAndStd()
    norm_Transformers = getNormTrans(varMean, varStd, varIsoChannels)
    trainAugmenters = getTrainAugmenter(norm_Transformers)

    
    ds = TropicalCycloneDataset(trainSet, rootRawData, transforms = trainAugmenters, maxForecastTime = maxForecastTime, fillOutlier = True)
    inp, lab = ds[0]
    print(inp.size())
    print(len(ds))


    
    # test dataloader

    batch_size = 2
    num_workers = 2
    pwt = True

    train_dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True,  num_workers= num_workers,  persistent_workers= pwt)
    train_features, train_labels = next(iter(train_dataloader))
    print(train_features.size()) #torch.Size([2, 465, 33, 33])
