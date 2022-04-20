
import PIL
from PIL import Image
import PIL
from pathlib import Path
from PIL import UnidentifiedImageError

path = Path("D:\\image_spam_and_ham\\train\\").rglob("*.JPG")
for img_p in path:
    try:
        img = PIL.Image.open(img_p)
    except PIL.UnidentifiedImageError:
            print(img_p)


