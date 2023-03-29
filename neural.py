from stuff import *
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten
from keras.models import Sequential, load_model
import random
import datetime
import os

IMAGES_DIR = './data_small/imgs/' #должен быть trailing slash
TRAIN_CSV_PATH = './data_small/train.csv'
TEST_CSV_PATH = './data_small/test.csv'
X_RESOLUTION = 512
Y_RESOLUTION = 128

CURRENT_MODEL_NAME = "model"
CURRENT_MODEL_PATH = os.path.join('./model', CURRENT_MODEL_NAME)
USE_SAVED_MODEL = True

ALL_IMAGES_COUNT = 122297
CHUNK_SIZE = 5000

if not USE_SAVED_MODEL:
    
    if os.path.isdir(CURRENT_MODEL_PATH):
        res = input(f"trained model named \"{CURRENT_MODEL_NAME}\" already exists and will be overwriten. Continue ? (y/n)")
        if res.lower() != "y":  
            exit()                  
    else:
        os.mkdir(CURRENT_MODEL_PATH)        
    
    #берем первые 2000 для авто теста модели

    current_chunk = 0
    chunk_count = ALL_IMAGES_COUNT // CHUNK_SIZE + 1
    
     
    print('Building neural network model...')
    model = Sequential()


    x_train, y_train = loadTrainData(IMAGES_DIR, TRAIN_CSV_PATH, X_RESOLUTION, Y_RESOLUTION)
    

    model.add(Dense(256, activation='relu', input_shape=x_train[0].shape))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    print('Training neural network...')
    
    
    test_count = 2000
    
    x_test = np.array(x_train[:test_count])
    y_test = np.array(y_train[:test_count])
    
    
    while(current_chunk < chunk_count):
        low = test_count + current_chunk * CHUNK_SIZE
    
    x_train = x_train[2000:]
    y_train = y_train[2000:]
    
    model.fit(x_train, y_train, epochs=50 )
    
    evalResult = model.evaluate(x_test, y_test)
    print(f'Model evaluate results: loss={evalResult[0]}     accuracy={evalResult[1]}')
    model.save(CURRENT_MODEL_PATH)
else:
    print('Loading neural network model...')
    model = load_model(CURRENT_MODEL_PATH)


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
