import pytesseract
from PIL import Image

# 首先使用PIL打开要识别的图像
img = Image.open('111.png')

# 转换图像到灰度模式
img = img.convert('L')

# 对图像进行二值化处理 (适用于白底黑字)
threshold = 150 # 设置阈值
img = img.point(lambda x: 0 if x < threshold else 255)

# 调用Tesseract进行识别
text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')

# 输出识别结果
print(text)