from io import BytesIO
from PIL import Image
import numpy as np

def formatImage(origin, targetRes = (512, 128)):
    img = Image.open(origin)
    size = img.size
    targetRel = targetRes[0] / targetRes[1]
    
    rel = size[0] / size[1]
    
    if rel > targetRel:
        w = size[0]
        h = rel / targetRel * size[1]
    else:
        h = size[1]
        w = targetRel / rel * size[0]
    
    img = img.crop((0, 0, w, h))
    img = img.resize(targetRes)
    img = img.convert("L")
    
    return img.getdata()