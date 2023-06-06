import cv2
from PIL import Image

__if_show_img = True

def read_img(path):
    global __if_show_img
    
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if __if_show_img:
        Image.fromarray(img).show()
        __if_show_img = False

    return img