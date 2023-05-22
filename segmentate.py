import numpy as np
import cv2
import matplotlib.pyplot as plt


"""Segmentate image to get rid of background noise
Read this to improve: https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
"""
def segmentate(img, num_objects=2):

    #issue: how to get rid of the shadow? maybe use a different color space?
    img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (7, 7), 5)
    # What is the better threshold value?
    ret, thresh = cv2.threshold(img_gray, 200, 255, 0)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    bounding_boxes = []
    
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < 500: # eliminates small contours
            continue
        if cv2.contourArea(contours[i]) > 50000: # eliminates unreasonably big contours
            continue
        if (len(contours[i]) > 8000): # this is a hack to eliminate the shadow
            continue
        bounding_boxes.append(((min(contours[i][:, :, 0])[0], min(contours[i][:, :, 1])[0]), 
                  (max(contours[i][:, :, 0])[0], max(contours[i][:, :, 1])[0]))) # (x1, y1), (x2, y2)
        if (len(bounding_boxes) == num_objects): # we only need two bounding boxes for two objects (may increase more)
            break
        
    processed_img = img.copy()
    for box in bounding_boxes:
        cv2.rectangle(processed_img, box[0], box[1], (0, 255, 0), 7)
    return (processed_img, bounding_boxes)

# use compare hist to count
    
    
# for i in range(1, 28):
#     name = 'data/DataImgLego/Second_batch/1 ({}).jpg'.format(i)
#     img = cv2.imread(name)
#     plt.subplot(121); plt.imshow(img); plt.title('Original')
#     plt.subplot(122); plt.imshow(segmentate(img), cmap='gray'); plt.title('Segmentated')
#     plt.show()