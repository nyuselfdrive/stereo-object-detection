'''

This is the stereo object detection demo.

from https://rdmilligan.wordpress.com/2016/05/23/disparity-of-stereo-images-with-python-and-opencv/

Most helpful tutorial: http://timosam.com/python_opencv_depthimage

make sure to install opencv-contrib-python as well

TODO: Convert to C++ and write python wrappers to take advantage of GPU
https://stackoverflow.com/questions/12957492/writing-python-bindings-for-c-code-that-use-opencv
'''
from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
from darknet import darknet
import matplotlib.pyplot as plt

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


netMain = None
metaMain = None
altNames = None


# YOLO parameters
config="cfg/igvc-yolov3.cfg"
weights="igvc-yolov3_6000.weights"
names="data/igvc.names"

netMain = darknet.load_net_custom(config.encode(
            "ascii"), weights.encode("ascii"), 0, 1)  # batch size = 1
#metaMain = darknet.load_meta(data.encode("ascii"))
altNames = [x.strip() for x in open(names, 'r').readlines()]

def get_distance(depth_val):
    '''
    Old get_distance function. 
    '''
    return 250.477*depth_val**(-1.1504)



# Create stereo matcher
window_size = 17
left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=160, # has to be divisible by 16
    blockSize=5,
    P1=8 * 3 * window_size ** 2,  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# Filter Parameters
lmbda = 8000.0
sigma = 1.2
 
# Weighted least squares filter
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)

wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

# Load camera calibration parameters
undistortion_maps_l = np.load("calibration/zed/undistortion_map_left.npy")
undistortion_maps_r = np.load("calibration/zed/undistortion_map_right.npy")
rectification_maps_l = np.load("calibration/zed/rectification_map_left.npy")
rectification_maps_r = np.load("calibration/zed/rectification_map_right.npy")


# Calibration parameters from zed calibration files. In my experience they didn't work very well
# mtx_l = np.load("camera_matrix_left.npy")
# mtx_r = np.load("camera_matrix_right.npy")
# dist_l = np.load("dist_coeffs_left.npy")
# dist_r = np.load("dist_coeffs_right.npy")
# newmtx_l, roi_l = cv2.getOptimalNewCameraMatrix(mtx_l,dist_l,(672,376),0)
# newmtx_r, roi_r = cv2.getOptimalNewCameraMatrix(mtx_r,dist_r,(672,376),0)

cap = cv2.VideoCapture(0)

darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        start = time.time()

        # Split frame into left and right images since the ZED camera 
        #  gives us both images side by side
        # left = frame[0:, 0:frame.shape[1]//2]
        # right = frame[0:, frame.shape[1]//2:]

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow(image)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        continue



        # Convert images to grayscale. Yields slight performance improvement 
        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Rectify images
        rect_l = cv2.remap(gray_l, undistortion_maps_l,
                        rectification_maps_l,
                        cv2.INTER_LINEAR)

        rect_r = cv2.remap(gray_r, undistortion_maps_r,
                        rectification_maps_r,
                        cv2.INTER_LINEAR)

        # Rectify with zed calibration
        # rect_l = cv2.undistort(gray_l, mtx_l, dist_l, None, newmtx_l)
        # rect_r = cv2.undistort(gray_r, mtx_r, dist_r, None, newmtx_r)
        
        # compute disparity maps
        disp_l = left_matcher.compute(rect_l, rect_r).astype(np.int16) #.astype(np.float32) / 16.0
        disp_r = right_matcher.compute(rect_r, rect_l).astype(np.int16) #.astype(np.float32) / 16.0

        # Post-filtering with Weighted Least Squares to smooth image
        filteredImg = wls_filter.filter(disp_l, rect_l, None, disp_r, None, rect_r).astype(np.float32)/16.0 
        
        # Optional blurring step to smoothen image
        # filteredImg = cv2.medianBlur(filteredImg,5)


        # Constant to convert distance into feet
        # D = b*f/d
        # D = Distance of point in real world
        # b = distance between cameras
        DIST_CONST = 2600/16#.90804598
        distance = (DIST_CONST / (avg_val+1))#*DIST_CONST
        
        # Display distance
        print("%d feet %d inches" % (int(distance), int((distance-int(distance))*12)))


        cmap = plt.get_cmap('jet')
        rgba_disparity = cmap(filteredImg/160)
        rgb_disparity = np.delete(rgba_disparity, 3, 2)
        h, w, c = rgb_disparity.shape
        resize_frame = cv2.resize(left, (w*2, h*2))
        cv2.rectangle(rgb_disparity, (x1, y1), (x2, y2), (0, 0, 0), 2)
        cv2.rectangle(left, (x1, y1), (x2, y2), (0, 0, 0), 2)
        # resize_disp = cv2.resize(rgb_disparity, (w, h))


        # Calculate FPS
        print("FPS: %0.2f" % (1.0/(time.time()-start)))
        
        
        both = np.hstack((left/255, rgb_disparity))
        # Calculate FPS
        #print("FPS: %0.2f" % (1.0/(time.time()-start)))

        cv2.imshow("Stereo depth viewer", both)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        continue
