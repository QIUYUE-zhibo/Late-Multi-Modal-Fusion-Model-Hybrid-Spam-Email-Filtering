import pytesseract
from PIL import Image
import os
import time


CPU_time=[]
img_path = 'D:\\test_images'
img_path_list = os.listdir(img_path)
start = time.perf_counter()
for img in img_path_list:
    img = os.path.join(img_path, img)
    image = Image.open(img)
    code = pytesseract.image_to_string(image, lang='eng')
    end = time.perf_counter()
    CPU_time.append(end - start)
print(CPU_time)
