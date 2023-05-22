import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def get_roi(img):
    width = img.shape[1]
    height = img.shape[0]
    roi_height = 100
    ret = img[(height>>1)- roi_height:(height>>1)+roi_height + 20 , 0:width]

    return ret