def defineLabel(fileName, maxForecastTime):

        # negative
        label = 0

        spltName = fileName.split('/')

        if spltName[0] == 'POSITIVE':
            label = 1
        elif int(spltName[1].split('_')[2]) <=  maxForecastTime // 3:
            label = 1

        return label
