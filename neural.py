from stuff import *
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential, load_model
import random
import datetime
import math

IMAGES_DIR = './data_small/imgs/' #должен быть trailing slash
TRAIN_CSV_PATH = './data_small/train.csv'
TEST_CSV_PATH = './data_small/test.csv'
X_RESOLUTION = 200
Y_RESOLUTION = 50
USE_SAVED_MODEL = True
CHUNK_SIZE = 1000

if not USE_SAVED_MODEL:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    x_train, y_train = loadTrainData(IMAGES_DIR, TRAIN_CSV_PATH, X_RESOLUTION, Y_RESOLUTION)

    #берем первые 2000 для авто теста модели
    x_test = x_train[:2000]
    x_train = x_train[2000:]
    y_test = y_train[:2000]
    y_train = y_train[2000:]

    x_test = np.array(x_test) / 255
    y_test = np.array(y_test)
    shapeSample = np.array(x_train[0]) / 255

    print('Building neural network model...')
    model = Sequential()

    model.add(Dense(X_RESOLUTION * Y_RESOLUTION, activation='relu', input_shape=shapeSample.shape))
    model.add(Dense(32, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])


    numberOfChunks = math.ceil(len(x_train) / CHUNK_SIZE)
    print('Training neural network...')
    for i in range(numberOfChunks):
        print(f'Chunk {i+1} of {numberOfChunks}')
        startIndex = i * CHUNK_SIZE
        endIndex = startIndex + CHUNK_SIZE
        if endIndex > len(x_train):
           endIndex = len(x_train)

        x_train_chunk = np.array(x_train[startIndex:endIndex]) / 255
        y_train_chunk = np.array(y_train[startIndex:endIndex])

        model.fit(x_train_chunk, y_train_chunk, epochs=5)


    evalResult = model.evaluate(x_test, y_test)
    print(f'Model evaluate results: loss={evalResult[0]}     accuracy={evalResult[1]}')
    model.save('./model')
else:
    print('Loading neural network model...')
    model = load_model('./model')


testFileNames = loadTestData(TEST_CSV_PATH)


while 1:
    random.seed(datetime.datetime.now().timestamp())
    pictureIndex = random.randint(0, len(testFileNames) - 1)

    testImage = formatImage(IMAGES_DIR + testFileNames[pictureIndex], (X_RESOLUTION, Y_RESOLUTION))
    model_answer = model.predict(np.array([testImage]))[0][0] #TODO не понял почему надо два раза [0]

    ans = 'это рукописный текст' if model_answer > 0.5 else'это печатный текст'
    print(f'Ответ нейросети: {ans}')

    showImage = cv2.imread(IMAGES_DIR + testFileNames[pictureIndex]) #TODO нет защиты от того, файл не является картинкой
    plt.imshow(showImage)
    plt.axis('off')
    plt.show()
