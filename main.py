import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import roi
import segmentate
import isolate

# cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap = cv.VideoCapture('data/lego_on_belt_2.mp4')
# cap.set(cv.CAP_PROP_POS_MSEC, 21000)

while (True):
    ret, frame = cap.read()
    if ret == True:
        frame_roi = roi.get_roi(frame)
        detected, bounding_boxes = segmentate.segmentate(frame_roi)
        isolated = isolate.isolate(frame_roi, bounding_boxes)
        cv.imshow('bounding boxes', detected)
        cv.imshow('isolated', isolated)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break