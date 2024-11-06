import random
import os
from configs.configs_parser import load_config
from defineLabel import defineLabel


def save_data(save_path, file_name, data_list):
    with open(f"{save_path}/{file_name}", "w", encoding="utf-8") as f: 
        for dataName in data_list: 
            f.write(f"{dataName}\n")


if __name__ == "__main__":

    config = load_config("./configs/configs.yml")
    
    rootRawData = config['data']['rootRawData']
    rootSplitData = config['data']['rootSplitData']
    maxForecastTime = config['data']['maxForecastTime']
    weighted_inputNorm = config['data']['weighted_inputNorm']

    trainRatio, valRatio, testRatio = tuple(config['data']['splitRatio'])

    w = os.walk(rootRawData)

    pos_files = []
    neg_files = []


    for (dirpath, dirnames, filenames) in w:

        for filename in filenames:

            pth = f"{os.path.relpath(dirpath, rootRawData)}/{filename}"
            category = defineLabel(pth, maxForecastTime)

            if category == 1:
                pos_files += [pth]
            else:
                neg_files += [pth]

    random.shuffle(pos_files)
    random.shuffle(neg_files)

    # split

    trainPosSize = int(len(pos_files) * trainRatio // 10)
    trainNegSize = int(len(neg_files) * trainRatio // 10)

    trainSet = pos_files[:trainPosSize] + neg_files[:trainNegSize]
    random.shuffle(trainSet)



    valPosSize = int(len(pos_files) * valRatio // 10)
    valNegSize = int(len(neg_files) * valRatio // 10)


    valSet = pos_files[trainPosSize:trainPosSize+valPosSize] + neg_files[trainPosSize:trainPosSize+valNegSize]
    random.shuffle(valSet)



    testSet = pos_files[trainPosSize+valPosSize:] + neg_files[trainPosSize+valNegSize:]
    random.shuffle(testSet)




    # save

    save_data(rootSplitData, "train.txt", trainSet)

    save_data(rootSplitData, "val.txt", valSet)

    save_data(rootSplitData, "test.txt", testSet)
