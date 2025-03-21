import cv2
import numpy as np

# 读取两幅图像
img1 = cv2.imread('image3.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('image3.jpg', cv2.IMREAD_GRAYSCALE)

# 定义sift对象
sift = cv2.SIFT_create()

# 检测关键点和计算描述子
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# 创建匹配器对象BFMatcher
bf = cv2.BFMatcher()

# 在第一幅图像中对每个关键点寻找与之匹配的特征点
matches = bf.knnMatch(des1, des2, k=2)

# 通过KNN筛选最佳匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# 提取匹配对应的关键点在源图像和目标图像上的坐标
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 使用RANSAC算法进行特征点匹配，排除错误匹配和重复特征点
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matches_mask = np.ravel(mask).tolist()
good_matches = [m for i, m in enumerate(good_matches) if matches_mask[i] == 1]

# 绘制匹配结果
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

# 显示结果
cv2.imshow("Matching result", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()

