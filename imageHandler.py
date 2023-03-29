from PIL import Image

def formatImage(origin, targetRes = (512, 128)):
    img = Image.open(origin)
    formatt = img.format
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
    

    # pixels = img.getdata()
    # width, height = img.size
    # matrix = []
    # i = 0
    
    # for x in range(width):
    #     matrix.append([])
    #     for y in range(height):
    #         pixel = pixels[i] / 255
    #         matrix[x].append(pixel)
    #         i = i + 1
    #pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    return img.getdata()

    
    # temp = BytesIO()    
    # img.save(temp, formatt)
    # img.close()
    
    # imgBytes = temp.getvalue()
    # temp.close()
    
    # return imgBytes