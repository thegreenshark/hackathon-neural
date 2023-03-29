#этот скрипт преобразует все картинки из INPUT_IMAGES_DIR, чтобы у всех было одинаковое разрешение, и сохраняет в OUTPUT_IMAGES_DIR, и чтобы сохранялся цвет фона
#на практике оказалось, на таких картинках нейросеть учится хуже, чем с черной заливкой сбоку

from PIL import Image
import os

#mode:
#0 = взять средний цвет нижней кромки
#1 = взять средний цвет правой кромки
def getAvgColor(mode, numberOfSamples):
    s = size[0] if mode == 0 else size[1]
    step = s // numberOfSamples
    if step == 0:
        step = 1

    sum = 0
    count = 0
    if mode == 0:
        for i in range(0, s, step):
            sum += img.getpixel((i, size[1] - 1))
            count += 1
    else:
        for i in range(0, s, step):
            sum += img.getpixel((size[0] - 1, i))
            count += 1

    return round(sum / count)





INPUT_IMAGES_DIR = './data_small/imgs/'
OUTPUT_IMAGES_DIR = './data_small/imgs_converted/'
NUMBER_OF_COLOR_SAMPLES = 16
TARGET_X_RESOLUTION = 512
TARGET_Y_RESOLUTION = 128



targetRes = (TARGET_X_RESOLUTION, TARGET_Y_RESOLUTION)

if(not os.path.isdir(INPUT_IMAGES_DIR)):
    os.mkdir(INPUT_IMAGES_DIR)
if(not os.path.isdir(OUTPUT_IMAGES_DIR)):
    os.mkdir(OUTPUT_IMAGES_DIR)



for filename in os.listdir(INPUT_IMAGES_DIR):
    filePath = os.path.join(INPUT_IMAGES_DIR, filename)
    if os.path.isfile(filePath):
        img = Image.open(filePath)
        img = img.convert("L") #перевод в ЧБ

        size = img.size
        targetRel = targetRes[0] / targetRes[1]

        rel = size[0] / size[1]

        avgColor = 0
        if rel > targetRel: #шире чем надо
            w = size[0]
            h = round(rel / targetRel * size[1])
            avgColor = getAvgColor(mode=0, numberOfSamples=NUMBER_OF_COLOR_SAMPLES)
            img = img.crop((0, 0, w, h)) #увеличиваем ширину или высоту для достижения нужного соотношения сторон
            img.paste(avgColor, [0, size[1], size[0], h]) #заливаем цветом область снизу от картинки

        else: #уже чем надо
            h = size[1]
            w = round(targetRel / rel * size[0])
            avgColor = getAvgColor(mode=1, numberOfSamples=NUMBER_OF_COLOR_SAMPLES)
            img = img.crop((0, 0, w, h)) #увеличиваем ширину или высоту для достижения нужного соотношения сторон
            img.paste(avgColor, [size[0], 0, w, size[1]]) #заливаем цветом область справа от картинки

        img = img.resize(targetRes)
        img.save(OUTPUT_IMAGES_DIR + filename)

