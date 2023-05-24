import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import roi
import segmentate
import isolate
import speed_infer
import time
import argparse
import identify_color
from kalman_filter import KalmanFilter

# to run realtime, call python main.py --realtime 1 --hístogram 0
# otherwise, for testing just call python main.py
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--debug', type=int, default=0,
                    help='debug mode')
parser.add_argument('--video', type=str, default='data/speed_high.mp4',
                    help='video path')
parser.add_argument('--realtime', type=int, default=0, help='realtime mode')
parser.add_argument('--threshold', type=int, default=160, help='threshold value for segmentate')
parser.add_argument('--histogram', type=int, default=1, help='use histogram to count')
args = parser.parse_args()

# cap = cv.VideoCapture(0, cv.CAP_DSHOW)
cap = cv.VideoCapture(args.video)
if (args.realtime == 1):
    cap = cv.VideoCapture(0, cv.CAP_DSHOW)
# cap.set(cv.CAP_PROP_POS_MSEC, 21000)

prev_frame_roi = None
previous_bounding_boxes = []
previous_isolated = None
prev_time = time.time()
prev_speed = 0.0

list_speed = []
my_filter = KalmanFilter()

cnt = 0
while (True):
    ret, frame = cap.read()
    if ret == True:
        frame_roi = roi.get_roi(frame)
        detected, bounding_boxes = segmentate.segmentate(args, frame_roi.copy())
        isolated = isolate.isolate(args, frame_roi, bounding_boxes)
        colors = identify_color.detect_color(args, frame_roi, bounding_boxes)
        for (box, color) in zip(bounding_boxes, colors):
            cv.putText(detected, color, box[0], cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv.LINE_AA)
        
        cur_time = time.time()
        time_elapsed = cur_time - prev_time
        if (args.realtime == 0):
            time_elapsed = 1.0/32.0
        # speed = speed_infer.get_speed(bounding_boxes, previous_bounding_boxes, cur_time - prev_time)
        # speed = speed_infer.get_speed_isolated(isolated, previous_isolated, time_elapsed, debug=debug)
        speed = speed_infer.optical_flow(frame_roi, prev_frame_roi, \
                previous_bounding_boxes, time_elapsed)
        if (speed == -1 or speed == 0.0):
            speed = prev_speed
        
        if (speed > 0):
            my_filter.state_update(speed)
            list_speed.append(speed)
            prev_speed = speed
        prev_time = cur_time
        previous_isolated = isolated
        prev_frame_roi = frame_roi
        previous_bounding_boxes = bounding_boxes
        
        cv.putText(detected, 'Speed: {:.2f} m/s'.format(my_filter.current_velocity), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
        cv.imshow('bounding boxes', detected)
        cv.imshow('isolated', isolated)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        previous_bounding_boxes = bounding_boxes
    else:
        break
    
variance = np.var(list_speed)
print(variance)
    
if (args.histogram == 1):
    plt.hist(list_speed, bins=np.arange(0.020, 0.40, 0.02))
    plt.title('Histogram of speed')
    plt.show()