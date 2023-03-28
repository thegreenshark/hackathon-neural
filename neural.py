from stuff import *
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential, load_model
import random
import datetime

IMAGES_DIR = './data_small/imgs/' #должен быть trailing slash
TRAIN_CSV_PATH = './data_small/train.csv'
TEST_CSV_PATH = './data_small/test.csv'
USE_SAVED_MODEL = False



if not USE_SAVED_MODEL:
    x_train, y_train = loadTrainData(IMAGES_DIR, TRAIN_CSV_PATH)

    print('Building neural network model...')
    model = Sequential()

    model.add(Dense(2, activation='relu', input_shape=x_train[0].shape))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    print('Training neural network...')
    model.fit(x_train, y_train, epochs=50 )

    model.save('./model')
else:
    print('Loading neural network model...')
    model = load_model('./model')


testFileNames = loadTestData(TEST_CSV_PATH)


while 1:
    random.seed(datetime.datetime.now().timestamp())
    pictureIndex = random.randint(0, len(testFileNames) - 1)

    testImage = cv2.imread(IMAGES_DIR + testFileNames[pictureIndex]) #TODO нет защиты от того, файл не является картинкой
    testImage = cv2.cvtColor(testImage, cv2.COLOR_BGR2GRAY) #перевод в чернобелое
    testImage = cv2.resize(testImage, (512, 128)) #TODO

    model_answer = model.predict(np.array([testImage]))[0][0] #TODO не понял почему надо два раза [0]

    ans = 'это рукописный текст' if model_answer > 0.5 else'это печатный текст'
    print(f'Ответ нейросети: {ans}')

    plt.imshow(testImage, cmap='binary')
    plt.axis('off')
    plt.show()
