import math

import cv2

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

img = cv2.imread('1.png') #读入图片
img_gray = cv2.imread('1.png', 0) #对图片并进行预处理：灰度转换

img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1) #对图片进行高斯滤波处理


edges = cv2.Canny(img_blur, 50, 200) #对图片进行canny边缘检测

cv2.imshow('canny', edges) #展示边缘检测的图
cv2.waitKey(0)
cv2.destroyAllWindows()

kernel = np.ones((2, 2), np.uint8)
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
edges_close = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, rectKernel) #闭操作
edges_dilate = cv2.dilate(edges_close, kernel, iterations=3) #膨胀操作
#由于我们下一步需要筛选出最外层边界，过多的边界信息会对筛选造成困难，所以加以闭操作和膨胀操作



contours, hierarchy = cv2.findContours(edges_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#找出图中的轮廓值，得到的轮廓值都是嵌套格式的
contours = sorted(contours, key=lambda cnts: cv2.arcLength(cnts, True), reverse=True) #轮廓排序

img_copy = img.copy()
res = cv2.drawContours(img_copy, contours, 0, (0, 0, 255), 2) #轮廓绘制
img_copy = img.copy()
cnt = contours[0]
epsilon = 0.03 * cv2.arcLength(cnt, True)  # epsilon占周长的比例
approx = cv2.approxPolyDP(cnt, epsilon, True) #对曲线进行采样，在曲线上取有限个点，将其变为折线
res2 = cv2.drawContours(img_copy, [approx], -1, (0, 255, 0), 5) #轮廓绘制

cv2.imshow('res2', res2)
cv2.waitKey(0)
cv2.destroyAllWindows()

[[lt], [lb], [rb], [rt]] = approx
[ltx, lty] = lt
[lbx, lby] = lb
[rbx, rby] = rb
[rtx, rty] = rt

lt = (ltx, lty)
lb = (lbx, lby)
rb = (rbx, rby)
rt = (rtx, rty)




# 仿射变换
width = max(math.sqrt((rtx - ltx) ** 2 + (rty - lty) ** 2), math.sqrt((rbx - lbx) ** 2 + (rby - lby) ** 2))
height = max(math.sqrt((ltx - lbx) ** 2 + (lty - lby) ** 2), math.sqrt((rtx - rbx) ** 2 + (rty - rby) ** 2))
pts1 = np.float32([[ltx, lty], [rtx, rty], [lbx, lby], [rbx, rby]]) #表示四个角点在原图中的坐标
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]]) #表示四个角点对应目标位置的坐标
M = cv2.getPerspectiveTransform(pts1, pts2) #对图像进行透视变换
width = int(width)
height = int(height)
dst = cv2.warpPerspective(img, M, (width, height)) #对图片进行视角转换

plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()
print(dst)
resu = cv2.threshold(dst, 120, 255, cv2.THRESH_BINARY)[1] #对图片进行二值化处理
plt.imshow(resu), plt.title('Result')
plt.show()
cv2.imwrite('OCR.jpg', resu)

