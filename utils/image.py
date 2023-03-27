import cv2
import numpy as np


def LoadImage(filename: str):
    return cv2.imread(filename)


def LoadImageGrayscale(filename: str):
    img = cv2.imread(filename)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def FlattenAndScaleImage(img: np.ndarray):
    img = img.ravel().astype("float32")
    img /= 255.0
    return img

