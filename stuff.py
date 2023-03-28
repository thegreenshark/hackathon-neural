import numpy as np
import cv2
import os



def loadTrainData(imgDir, trainCsvPath, imgXres, imgYres):
    print('Reading training csv file...')
    trainFile = open(trainCsvPath, mode='r', encoding='utf-8')
    trainFileLines = trainFile.readlines()
    trainFile.close()

    x_train = []
    y_train = []

    total = len(trainFileLines[1:])
    progressStep = total // 100
    count = 0

    print('Loading training images...')
    for line in trainFileLines[1:]:
        splitLine = line.split(',')
        fileName = splitLine[-2]
        isHandWritten = int(splitLine[-1])

        #в текстах могут быть запятые, тогда split разбивает текст, а нам этого не надо
        # text = splitLine[0]
        # for k in range(1, len(splitLine) - 2):
        #     text += splitLine[k]

        # if len(text) > 0 and text[0] == '"':
        #     text = text[1:]
        # if len(text) > 0 and text[-1] == '"':
        #     text = text[:-1]


        filePath = imgDir + fileName
        if os.path.isfile(filePath):
            image = cv2.imread(filePath) #TODO нет защиты от того, файл не является картинкой
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #перевод в чернобелое
            image = cv2.resize(image, (imgXres, imgYres)) #TODO
            x_train.append(image)
            y_train.append(isHandWritten)



        if count % progressStep == 0:
            print(f'{round(count / total * 100)}%', end="\r")
        count += 1
    print('100%')

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    x_train = x_train / 255

    return x_train, y_train



def loadTestData(testCsvPath):
    print('Reading test csv file...')
    testFile = open(testCsvPath, mode='r', encoding='utf-8')
    testFileLines = testFile.readlines()
    testFile.close()

    testFileNames = []
    for line in testFileLines[1:]:
        fileName = line.rstrip()
        testFileNames.append(fileName)
    #сами картинки тут не загружаем, чтобы не расходовать память почем зря

    return testFileNames
