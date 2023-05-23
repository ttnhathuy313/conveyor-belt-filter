import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import roi
import segmentate
import isolate
import speed_infer
import time

# cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap = cv.VideoCapture('data/lego_on_belt_2.mp4')
# cap.set(cv.CAP_PROP_POS_MSEC, 21000)

prev_frame_roi = None
previous_bounding_boxes = []
previous_isolated = None
prev_time = time.time()
prev_speed = 0.0

cnt = 0
while (True):
    ret, frame = cap.read()
    if ret == True:
        frame_roi = roi.get_roi(frame)
        detected, bounding_boxes = segmentate.segmentate(frame_roi)
        isolated = isolate.isolate(frame_roi, bounding_boxes)
        
        cur_time = time.time()
        time_elapsed = cur_time - prev_time
        time_elapsed = 1.0/32.0
        # speed = speed_infer.get_speed(bounding_boxes, previous_bounding_boxes, cur_time - prev_time)
        # speed = speed_infer.get_speed_isolated(isolated, previous_isolated, time_elapsed, debug=debug)
        speed = speed_infer.optical_flow(frame_roi, prev_frame_roi, \
                previous_bounding_boxes, time_elapsed)
        if (speed == -1):
            speed = prev_speed
        
        if (speed != -1):
            prev_speed = speed
        prev_time = cur_time
        previous_isolated = isolated
        prev_frame_roi = frame_roi
        previous_bounding_boxes = bounding_boxes
        
        cv.putText(detected, 'Speed: {:.2f} m/s'.format(speed), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow('bounding boxes', detected)
        cv.imshow('isolated', isolated)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        previous_bounding_boxes = bounding_boxes
    else:
        break