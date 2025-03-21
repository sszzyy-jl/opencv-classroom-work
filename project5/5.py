import numpy as np
import argparse
import cv2
img = cv2.imread('1.png')
h, w = img.shape[:2]
# Display original image
cv2.imshow('Original', img)
cv2.waitKey(0)
#灰度转换
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0)

edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)