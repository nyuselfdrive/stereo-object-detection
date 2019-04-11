'''
from https://rdmilligan.wordpress.com/2016/05/23/disparity-of-stereo-images-with-python-and-opencv/

Most helpful tutorial: http://timosam.com/python_opencv_depthimage

make sure to install opencv-contrib-python as well

TODO: Convert to C++ and write python wrappers to take advantage of GPU
https://stackoverflow.com/questions/12957492/writing-python-bindings-for-c-code-that-use-opencv
'''

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt



def get_distance(depth_val):
    return 250.477*depth_val**(-1.1504)

# disparity settings

# stereo_orig = cv2.StereoSGBM_create(
#     minDisparity = min_disp,
#     numDisparities = num_disp,
#     # SADWindowSize = window_size,
#     uniquenessRatio = 10,
#     speckleWindowSize = 100,
#     speckleRange = 32,
#     disp12MaxDiff = 1,
#     P1 = 8*3*window_size**2,
#     P2 = 32*3*window_size**2,
#     # fullDP = False
# )

# stereo = cv2.StereoSGBM_create(
#     minDisparity = min_disp,
#     numDisparities = num_disp,
#     uniquenessRatio = 10,
#     speckleWindowSize = 100,
#     speckleRange = 32,
#     disp12MaxDiff = 1,
#     P1 = 8*3*window_size**2,
#     P2 = 32*3*window_size**2,
# )

window_size = 17

# left_matcher = cv2.StereoBM_create(numDisparities=80, blockSize=9)

left_matcher = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=160,             # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=5,
    P1=8 * 3 * window_size ** 2,    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=15,
    speckleWindowSize=0,
    speckleRange=2,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

# FILTER Parameters
lmbda = 8000.0
sigma = 1.2
 
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)

wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)

# Load camera calibration parameters
undistortion_maps_l = np.load("zed_out/undistortion_map_left.npy")
undistortion_maps_r = np.load("zed_out/undistortion_map_right.npy")
rectification_maps_l = np.load("zed_out/rectification_map_left.npy")
rectification_maps_r = np.load("zed_out/rectification_map_right.npy")


# Calibration parameters from zed calibration files
# mtx_l = np.load("camera_matrix_left.npy")
# mtx_r = np.load("camera_matrix_right.npy")
# dist_l = np.load("dist_coeffs_left.npy")
# dist_r = np.load("dist_coeffs_right.npy")
# newmtx_l, roi_l = cv2.getOptimalNewCameraMatrix(mtx_l,dist_l,(672,376),0)
# newmtx_r, roi_r = cv2.getOptimalNewCameraMatrix(mtx_r,dist_r,(672,376),0)

cap = cv2.VideoCapture(1)
i=1
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        start = time.time()

        # Split frame into left and right images
        left = frame[0:, 0:frame.shape[1]//2]
        right = frame[0:, frame.shape[1]//2:]

        # Convert images to grayscale. 
        gray_l = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        # Rectify images
        rectified_l = cv2.remap(gray_l, undistortion_maps_l,
                        rectification_maps_l,
                        cv2.INTER_LINEAR)

        rectified_r = cv2.remap(gray_r, undistortion_maps_r,
                        rectification_maps_r,
                        cv2.INTER_LINEAR)

        # Rectify with zed calibration
        # rectified_l = cv2.undistort(gray_l, mtx_l, dist_l, None, newmtx_l)
        # rectified_r = cv2.undistort(gray_r, mtx_r, dist_r, None, newmtx_r)
        
        # compute disparity maps
        disparity_l = left_matcher.compute(rectified_l, rectified_r).astype(np.int16) #.astype(np.float32) / 16.0
        disparity_r = right_matcher.compute(rectified_r, rectified_l).astype(np.int16) #.astype(np.float32) / 16.0

        
        # displ = np.int16(disparity_l)
        # dispr = np.int16(disparity_r)


        # Post-filtering with Weighted Least Squares to smooth image
        filteredImg = wls_filter.filter(disparity_l, rectified_l, None, disparity_r, None, rectified_r).astype(np.float32)/16.0  # important to put "rectified_l" here!!!
        filteredImg = cv2.medianBlur(filteredImg,5)
        # print("Max", filteredImg.max()*16)
        # filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
        # filteredImg = np.uint8(filteredImg)

        # medianBlur_rgba = cmap(medianBlur)
        # medianBlur_rgb = np.delete(medianBlur_rgba, 3, 2)
        h, w = filteredImg.shape
        y1, y2, x1, x2 = h//2+40-15, h//2+40+15, w//2-25,w//2+25
        center_square = filteredImg[y1:y2, x1:x2]
        # cv2.imshow("center_square", center_square/160)
        avg_val = np.mean(center_square)

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
        
        
        both = np.hstack((left/255, rgb_disparity))
        # Calculate FPS
        #print("FPS: %0.2f" % (1.0/(time.time()-start)))

        cv2.imshow("Stereo depth viewer", both)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        if key & 0xFF == ord('p'):
            cv2.imwrite("drum-%d.png" % i, filteredImg)
            i+=1
        continue
        
        # Scale the values according to our settings
        disparity_disp = (disparity-min_disp)/num_disp
        
        # Blur the image to get rid of noise. Choose your favorite blur method
        cmap = plt.get_cmap('jet')

        # Bilateral filtering
        bilateralBlur = cv2.bilateralFilter(disparity_disp,15,25,25)
        bilateralBlur_rgba = cmap(bilateralBlur)
        bilateralBlur_rgb = np.delete(bilateralBlur_rgba, 3, 2)

        # Gaussian blur
        # gaussianBlur = cv2.GaussianBlur(disparity_disp,(5,5),0)
        # gaussianBlur_rgba = cmap(gaussianBlur)
        # gaussianBlur_rgb = np.delete(gaussianBlur_rgba, 3, 2)

        # Median blur
        # medianBlur = cv2.medianBlur(disparity_disp,5)
        # medianBlur_rgba = cmap(medianBlur)
        # medianBlur_rgb = np.delete(medianBlur_rgba, 3, 2)


        # Convert disparity map to RGB heatmap
        # cmap = plt.get_cmap('jet')
        # rgba_disparity = cmap(disparity_rect_disp_orig)
        # rgb_disparity = np.delete(rgba_disp_orig, 3, 2)
        
        # both_disp = np.hstack((left/255, gaussianBlur_rgb, medianBlur_rgb))

        h, w, c = left.shape
        result = cv2.resize(bilateralBlur_rgb, (w*2, h*2))
        cv2.imshow("Stereo depth viewer", result)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        # Calculate FPS
        print("FPS: %0.2f" % (1.0/(time.time()-start)))
        

   
