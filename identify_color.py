import cv2 as cv
import numpy as np 

color_map = dict({
    'yellow': (83, 104, 118),
    'red': (55, 54, 134),
    'black': (50, 40, 42),
    'green':(53, 110, 24),
})


def detect_color(args, img, boxes):
    ret = []
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    for box in boxes:
        x1, y1 = box[0]
        x2, y2 = box[1]
        roi = img[y1:y2, x1:x2]
        roi_gray = img_gray[y1:y2, x1:x2]
        mask = cv.threshold(roi_gray, args.threshold, 255, cv.THRESH_BINARY)[1]
        colors = roi[mask == 0]
        avg_color = np.mean(colors, axis=0)
        if (args.debug):
            print(avg_color)
            color_img = np.full(roi.shape, avg_color, dtype=np.uint8)
            cv.imshow('color', color_img)
            cv.imshow('roi', roi)
            cv.imshow('mask', mask)
            cv.waitKey(0)
        

        min_dist = 100000000
        fit_color = '?'
        for color in color_map:
            if (np.linalg.norm(avg_color - color_map[color]) < min_dist):
                min_dist = np.linalg.norm(avg_color - color_map[color])
                fit_color = color
        ret.append(fit_color)
    return ret