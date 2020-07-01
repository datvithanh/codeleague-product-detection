import math

import cv2
import numpy as np
import imutils

def load_image(x, path=True):
    if path == True:
        x = cv2.imread(x)

    # x = imutils.rotate_bound(x, angle)
    # x = cv2.resize(x, (int(224), int(224)))
    
    # x = x.astype('float')/255 - [0.485, 0.456, 0.406]
    # x = x / [0.229, 0.224, 0.225]
    # x = np.array(x).transpose(2,0,1)

    return x

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.01, sh = 0.05, r1 = 0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):
        if np.random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]
           
                target_area = np.random.uniform(self.sl, self.sh) * area
                aspect_ratio = np.random.uniform(self.r1, 1/self.r1)
    
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
    
                if w < img.shape[1] and h < img.shape[0]:
                    x1 = np.random.randint(0, img.shape[0] - h)
                    y1 = np.random.randint(0, img.shape[1] - w)

                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                    img[x1:x1+h, y1:y1+w, 1] = self.mean[1]
                    img[x1:x1+h, y1:y1+w, 2] = self.mean[2]

        return img


# Tensor
class Cutout(object):
    def __init__(self, length=50):
        self.length = length

    def __call__(self, img):
        h, w = img.shape[0], img.shape[1]    # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)
        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = np.expand_dims(mask, 2)
        img = img * mask
        return img

class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """
    def __call__(self, x):
        x = cv2.resize(x, (int(400), int(400)))
    
        x = x.astype('float')/255 - [0.485, 0.456, 0.406]
        x = x / [0.229, 0.224, 0.225]

        x = np.array(x)[:,:,::-1].transpose(2,0,1)
        return x


class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""
    def __call__(self, image_x):
        p = np.random.random()
        if p < 0.3:
            new_image_x = cv2.flip(image_x, 1)
            return new_image_x
        else:
            return image_x



