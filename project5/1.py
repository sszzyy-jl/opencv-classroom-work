import cv2
import numpy as np

# 读取待处理图片文件
img = cv2.imread('1.png')

# 将图片转为灰度图像
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 检测边缘，使用Canny算子
edges = cv2.Canny(gray_img,50,150,apertureSize = 3)

# 寻找轮廓
contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# 找到最大轮廓
max_area = 0
max_cnt = None
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > max_area:
        max_area = area
        max_cnt = cnt

# 获取顶点坐标，并绘制矩形框
rect = cv2.minAreaRect(max_cnt)
points = cv2.boxPoints(rect)
points = np.int32(points)
cv2.drawContours(img,[points],0,(0,0,255),2)

# 计算目标大小、位置，并进行透视变换
width = int(rect[1][0])
height = int(rect[1][1])
src_pts = np.float32(points)
dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype=np.float32)
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(img, M, (width, height))

# 显示结果
cv2.imshow('Original Image', img)
cv2.imshow('Warped Image', warped)
cv2.waitKey(0)