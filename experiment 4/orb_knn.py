import cv2
from matplotlib import pyplot as plt

# Load the images.

img0 = cv2.imread('1.jpg',cv2.COLOR_BGR2GRAY)
img1 = cv2.imread('2.jpg',cv2.COLOR_BGR2GRAY)

# Perform ORB feature detection and description.

orb = cv2.ORB_create()
kp0, des0 = orb.detectAndCompute(img0, None)
kp1, des1 = orb.detectAndCompute(img1, None)

# Perform brute-force KNN matching.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
pairs_of_matches = bf.knnMatch(des0, des1, k=2)

# Sort the pairs of matches by distance.
pairs_of_matches = sorted(pairs_of_matches, key=lambda x:x[0].distance)

# Apply the ratio test.
matches0 = [x[0] for x in pairs_of_matches
           if len(x) > 1 and x[0].distance < x[1].distance and x[0].distance < 0.8 * x[1].distance]
matches1 = [x[0] for x in pairs_of_matches
           if len(x) > 1 and x[0].distance > 0.8 * x[1].distance]

# Draw the best 25 matches.
img_matches0 = cv2.drawMatches(
    img0, kp0, img1, kp1, matches0[:50], img1,matchColor=[0,255,0],
    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
img_matches1 = cv2.drawMatches(
    img0, kp0, img1, kp1, matches1[:100], img1,matchColor=[255,0,0],
    flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# Show the matches.
plt.imshow(img_matches0+img_matches1)
plt.show()
