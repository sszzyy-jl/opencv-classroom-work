import cv2
import numpy as np

# 读取原始图像
img = cv2.imread("1.png")

# 使用Canny算法寻找边缘（调整下面两个参数以适应不同的图片）
edges = cv2.Canny(img, 50, 150)

# 寻找轮廓，并选取外轮廓
contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 找到最大的轮廓，并获取其矩形区域（即小票区域）
max_contour = max(contours, key=cv2.contourArea)
rect = cv2.minAreaRect(max_contour)
box = np.int0(cv2.boxPoints(rect))

# 将小票区域用绿色矩形框起来
cv2.drawContours(img, [box], -1, (0, 255, 0), 2)

# 定义原始图像中小票四个顶点位置（注意顺序）
src_points = np.float32([box[1], box[2], box[0], box[3]])

# 定义目标图像中小票四个顶点位置，通过计算可得
dst_points = np.float32([ [100, 600],[100, 200],  [400, 200],[400, 600]])

# 计算透视变换矩阵
M = cv2.getPerspectiveTransform(src_points, dst_points)

# 进行透视变换
result_img = cv2.warpPerspective(img, M,(img.shape[1], img.shape[0]))

# 显示结果图像
cv2.imshow("Result", result_img)
cv2.waitKey(0)