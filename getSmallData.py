#этот скрипт рандомно выбирает указанное количество картинок из датасета и помещает в ./data_small/
#чтобы при работе с мелким датасетом побыстрее всё грузилось

import os
import random
import datetime
import shutil


TRAIN_IMAGES_COUNT = 20000
TEST_IMAGES_COUNT = 5000

random.seed(datetime.datetime.now().timestamp())



trainFile = open('./data/train.csv', mode='r', encoding='utf-8')
trainFileLines = trainFile.readlines()
trainFile.close()



trainOutLines = []
trainOutLines.append(trainFileLines[0])

for j in range(TRAIN_IMAGES_COUNT):
    lineIndex = random.randint(1, len(trainFileLines) - 1)
    while (trainFileLines[lineIndex] in trainOutLines): #не повторяемся
        lineIndex = random.randint(1, len(trainFileLines) - 1)

    trainOutLines.append(trainFileLines[lineIndex])

trainFile_small = open('./data_small/train.csv', mode='w', encoding='utf-8')
trainFile_small.writelines(trainOutLines)
trainFile_small.close()



trainImgFilesNames = []

for line in trainOutLines[1:]:
    splitLine = line.split(',')
    trainImgFilesNames.append(splitLine[-2])

imgDirectiory = './data/imgs/'
for filename in os.listdir(imgDirectiory):
    filePath = os.path.join(imgDirectiory, filename)
    if os.path.isfile(filePath) and filename in trainImgFilesNames:
        shutil.copyfile(filePath, os.path.join('./data_small/imgs/', filename))








testFile = open('./data/test.csv', mode='r', encoding='utf-8')
testFileLines = testFile.readlines()
testFile.close()

testOutLines = []
testOutLines.append(testFileLines[0])

for j in range(TEST_IMAGES_COUNT):
    lineIndex = random.randint(1, len(testFileLines) - 1)
    while (testFileLines[lineIndex] in testOutLines): #не повторяемся
        lineIndex = random.randint(1, len(testFileLines) - 1)

    testOutLines.append(testFileLines[lineIndex])


testFile_small = open('./data_small/test.csv', mode='w', encoding='utf-8')
testFile_small.writelines(testOutLines)
testFile_small.close()



testImgFilesNames = []

for line in testOutLines[1:]:
    testImgFilesNames.append(line.rstrip())

imgDirectiory = './data/imgs/'
for filename in os.listdir(imgDirectiory):
    filePath = os.path.join(imgDirectiory, filename)
    if os.path.isfile(filePath) and filename in testImgFilesNames:
        shutil.copyfile(filePath, os.path.join('./data_small/imgs/', filename))