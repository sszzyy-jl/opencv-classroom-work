import cv2

background= cv2.imread("1.jpg")
submarine = cv2.imread("submarine.png")

# 裁剪潜水艇图片大小以适应背景图片
submarine= cv2.resize(submarine, (background.shape[1], background.shape[0]),
                       interpolation=cv2.INTER_AREA)
rows,cols,channels=submarine.shape
roi=background[0:rows,0:cols]
submarine = cv2.cvtColor(submarine, cv2.COLOR_RGB2BGR)

mask = cv2.cvtColor(submarine, cv2.COLOR_BGR2GRAY)
mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
mask_inv = cv2.bitwise_not(mask)

background_masked = cv2.bitwise_and(roi, roi, mask=mask)
submarine_masked = cv2.bitwise_and(submarine, submarine, mask=mask_inv)

final_image = cv2.add(background_masked, submarine_masked)
background[0:rows,0:cols]=final_image
background[0:rows,0:cols]=final_image

cv2.imshow("Merged Image", background)
cv2.waitKey(0)