import cv2
import numpy as np

# 读取待处理图片文件
img = cv2.imread('1.png')

# 将图片转为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 对灰度图像进行高斯模糊，去除噪点
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 进行Canny边缘检测
edges = cv2.Canny(blur, 50, 150, apertureSize=3)

# 进行霍夫直线变换，提取直线
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# 计算出四个边界点
points = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    points.append((x1, y1))
    points.append((x2, y2))

rect = cv2.minAreaRect(np.array(points))
box = cv2.boxPoints(rect)
box = np.int0(box)

# 绘制边界框和边缘检测结果
cv2.drawContours(img,[box],0,(0,255,0),2)
cv2.imshow('img', img)
cv2.waitKey(0)