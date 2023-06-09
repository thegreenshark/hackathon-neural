from PIL import Image

def formatImage(origin, targetRes = (512, 128)):
    img = Image.open(origin)
    size = img.size
    targetRel = targetRes[0] / targetRes[1]

    rel = size[0] / size[1]

    if rel > targetRel: #шире чем надо
        w = size[0]
        h = rel / targetRel * size[1]
    else: #уже чем надо
        h = size[1]
        w = targetRel / rel * size[0]

    img = img.crop((0, 0, w, h))
    img = img.resize(targetRes)
    img = img.convert("L")

    return img.getdata()




def openImage(origin):
    img = Image.open(origin)
    img = img.convert("L")
    return img.getdata()