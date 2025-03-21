import pytesseract
from PIL import Image

# 调用Tesseract进行识别
def ocr_image(path):
    img = Image.open(path)
    text = pytesseract.image_to_string(img, lang='eng')
    return text

# 读取待处理图片并进行OCR
text = ocr_image('2.png')

# 打印所有识别结果
print(text)
