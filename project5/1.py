import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取两幅图像，假设为left和right
left = cv2.imread('image.jpg')
right = cv2.imread('image3.jpg')

# 初始化SIFT检测器并提取特征点和特征描述符
sift = cv2.SIFT_create()
keypoints_left, descriptors_left = sift.detectAndCompute(left, None)
keypoints_right, descriptors_right = sift.detectAndCompute(right, None)

# 利用BFMatcher算法进行特征点匹配
matcher = cv2.BFMatcher()
matches = matcher.match(descriptors_left, descriptors_right)

# 根据SRNSAC随机采样一致算法筛选出最优的特征匹配点对
src_pts = np.float32([keypoints_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)

# 利用得到的投影映射矩阵计算变换后的左侧图像
h, w = left.shape[:2]
pts_left = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
pts_right = cv2.perspectiveTransform(pts_left, M)
pts = np.concatenate((pts_left, pts_right), axis=0)
[x_min, y_min] = np.int32(pts.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(pts.max(axis=0).ravel() + 0.5)
t = [-x_min, -y_min]
transform_matrix = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
result = cv2.warpPerspective(left, transform_matrix.dot(M), (x_max - x_min, y_max - y_min))

# 将两幅图像进行拼接
#result[t[1]:h+t[1], t[0]:w+t[0]] = right

# 显示拼接结果并保存到文件中
#cv2.imshow('result', result)
#cv2.imwrite('result.jpg', result)

#等待用户按下任意键退出
#cv2.waitKey(0)
#cv2.destroyAllWindows()
h, w =left.shape[:2]
aligned_image = cv2.warpPerspective(left, M, (w + right.shape[1], h))
aligned_image[0:right.shape[0], 0:right.shape[1]] = right

# 显示拼接结果。
plt.imshow(cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB))
plt.show()