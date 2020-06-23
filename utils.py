import cv2
import numpy as np
import imutils

def load_image(x, path=True, angle=0, gray=False):
    if path == True:
        x = cv2.imread(x)

    x = imutils.rotate_bound(x, angle)
    x = cv2.resize(x, (int(224), int(224)))
    
    if gray == True:
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x = np.stack((x,)*3, axis=-1) / 255 
    else:
        x = x.astype('float')/255 - [0.485, 0.456, 0.406]
        x = x / [0.229, 0.224, 0.225]
    x = np.array(x).transpose(2,0,1)

    return x


