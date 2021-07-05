import glob
import os
import random
import cv2


def blur(path):
    img = cv2.imread(path)
    row, col, ch = img.shape
    sz = random.randint(1,3)
    # print(sz)
    ksize = (sz, sz)
    # Using cv2.blur() method
    image = cv2.blur(img, ksize)
    return image


for file in glob.glob("/home/longhn/Anotation_1706/out_1006/*.jpg"):
    img = blur(file)
    txt = file.replace('.jpg','.txt')
    train = file.replace('out_1006','train')
    final = '/home/longhn/Anotation_1706/train/'
    mv = 'cp '+ txt +' '+final
    os.system(mv)
    cv2.imwrite(train,img)