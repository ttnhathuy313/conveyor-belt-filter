import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

fy = 5.5e-4 # focal length
D = 0.3 # distance from camera to legos
magic_number = fy/D # magic number to convert pixel to meter

#not robust to noise
def get_speed(cur_boxes, prev_boxes, time_elapsed):
    """Get the speed of the object in the image
    cur_boxes: a list of bounding boxes
    prev_boxes: a list of bounding boxes
    time_elapse: time elapse between two frames
    """
    if (len(cur_boxes) == 0 or len(prev_boxes) == 0):
        return -1
    dist = -1
    for box in cur_boxes:
        min_dist = 1000000
        id = -1
        for prev_box in prev_boxes:
            dist = np.linalg.norm(np.array(box[0]) - np.array(prev_box[0]))
            if (dist < min_dist):
                min_dist = dist
                id = prev_boxes.index(prev_box)
        if (abs(box[0][1] - prev_boxes[id][0][1]) > 30):
            continue
        dist = abs(box[0][1] - prev_boxes[id][0][1])
        break
    if (dist == -1):
        return -1
    
    return dist * magic_number / time_elapsed

# too slow to be used
def get_speed_isolated(cur_isolated, prev_isolated, time_elapsed, debug=0):
    """Get the speed of the object in the image using stereo system
    cur_isolated: the image of the isolated object
    prev_isolated: the image of the isolated object
    time_elapse: time elapse between two frames
    """
    if (debug == 1):
        print('wtf')
        plt.subplot(121); plt.imshow(prev_isolated, cmap='gray'); plt.title('Previous')
        plt.subplot(122); plt.imshow(cur_isolated, cmap='gray'); plt.title('Current')
        plt.show()
    if (prev_isolated is None):
        return -1
    
    window_size = 5 
    nDisp = 16

    imgL = cur_isolated
    imgR = prev_isolated
    left_matcher = cv.StereoSGBM_create(
        minDisparity=-1, # the disparity only goes in one direction
        numDisparities=nDisp, # each disparity value is multiplied with 16 to create better precision.
        blockSize=window_size, # the size of the block to compare
        P1=8 * 3 * window_size,
        P2=8 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=1,
        speckleWindowSize=5,
        speckleRange=5,
        preFilterCap=63,
        mode=cv.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv.ximgproc.createRightMatcher(left_matcher)
    displ = left_matcher.compute(imgL, imgR) #.astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  #.astype(np.float32)/16

    lmbda = 5000
    sigma = 3

    wls_filter = cv.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)
    mean = np.mean(filteredImg[filteredImg != 0])
    filteredImg[filteredImg == 0] = mean
    filteredImg = cv.normalize(src=filteredImg, dst=filteredImg, beta=1, alpha=255, norm_type=cv.NORM_MINMAX)
    if (debug == 1):
        print('what')
        plt.imshow(filteredImg, cmap='gray')
        plt.title('Refined disparity map')
        plt.show()
    return 0

def optical_flow(cur_img, prev_img, prev_bounding_boxes, time_elapsed):
    if (prev_img is None):
        return -1
    prev_img = cv.cvtColor(prev_img, cv.COLOR_BGR2GRAY)
    cur_img = cv.cvtColor(cur_img, cv.COLOR_BGR2GRAY)
    
    mask = np.zeros_like(prev_img)
    for box in prev_bounding_boxes:
        mask[box[0][1]:box[1][1], box[0][0]:box[1][0]] = 255
    feature_params = dict( maxCorners = 5,
                       qualityLevel = 0.3,
                       minDistance = 10,
                       blockSize = 5 )
    lk_params = dict( winSize  = (15,15),
                  maxLevel = 1,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.1))
    corners = cv.goodFeaturesToTrack(prev_img, mask = mask, **feature_params)
    if (corners is None):
        return -1
    next_corners, st, err = cv.calcOpticalFlowPyrLK(prev_img, cur_img, corners, None, **lk_params)
    if (next_corners is None):
        return -1
    good_old = corners[st==1]
    good_new = next_corners[st==1]
    #Draw corners
    sum = 0
    cnt = 0
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        dist = np.linalg.norm(np.array([a, b]) - np.array([c, d]))
        if (dist > 20):
            continue
        sum += dist
        cnt += 1
        prev_img = cv.line(prev_img, (int(a), int(b)), (int(c), int(d)), 255, 2)
    avg = 0
    if (cnt):
        avg = sum / cnt
    
    cv.putText(prev_img, 'Average distance: {:.2f}'.format(avg), (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    cv.imshow('Corners',prev_img)
    
    return avg * magic_number / time_elapsed