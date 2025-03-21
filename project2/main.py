import numpy as np
import cv2

minDisparity = 16
numDisparities = 192 - minDisparity

stereo = cv2.StereoSGBM_create(minDisparity=minDisparity,
                               numDisparities=numDisparities,
                               )


left_img = cv2.imread('11.jpg', cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread('22.jpg', cv2.IMREAD_GRAYSCALE)

disparity = stereo.compute(left_img, left_img).astype(np.float32) / 16.0

focal_length = 27
baseline = 0.1
depth = np.zeros_like(disparity)
depth[disparity > 0] = (baseline * focal_length) / disparity[disparity > 0]

depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow('Disparity', (disparity - minDisparity) / numDisparities)

cv2.imshow("Depth map", depth_norm)
cv2.waitKey()