import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def isolate(img, boxes):
    """Isolate an object from the background
    boxes: a list of bounding boxes
    """
    img_gray= cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    isolated = np.full(img_gray.shape, 255, dtype=np.uint8)
    for box in boxes:
        isolated[box[0][1]:box[1][1], box[0][0]:box[1][0]] = img_gray[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    # What is the better threshold value?
    ret, thresh = cv.threshold(isolated, 210, 255, cv.THRESH_BINARY)
    return thresh
