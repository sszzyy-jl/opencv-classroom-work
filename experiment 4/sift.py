import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('1.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('2.jpg')
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1,descriptors2,k=2)

goodMatchs = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        goodMatchs.append(m)

badMatchs = []
for m, n in matches:
    if m.distance > 0.7 * n.distance:
        badMatchs.append(m)

pic3 = cv2.drawMatches(img1=img1, keypoints1=keypoints1, img2=img2, keypoints2=keypoints2, matches1to2=goodMatchs,matchColor=[0,255,0],outImg=None)
pic4 = cv2.drawMatches(img1=img1, keypoints1=keypoints1, img2=img2, keypoints2=keypoints2, matches1to2=badMatchs,matchColor=[255,0,0],outImg=None)

plt.imshow(pic3+pic4)
plt.show()