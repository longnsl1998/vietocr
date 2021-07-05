import glob
import numpy as np
import os
import time
import cv2

def check_img_too_large(str_img_file):
    img_input = cv2.imread(str_img_file)
    if not isinstance(img_input, (np.ndarray)):
        return False

    if img_input.shape[0] < 12 or img_input.shape[1] < 12:
        return False
    wid = img_input.shape[1]
    hei = img_input.shape[0]
    if wid / hei > 30:
        return True  ### image too large, not use for training and testing
    else:
        return False
i = 0
count = 0
url_error = '/home/longhn/Annotation_2506/error/'
dir_error = 'mkdir ' + url_error

if os.path.exists(url_error) is False:
    os.system(dir_error)
for file in glob.glob("/home/longhn/Annotation_2506/data/*.txt"):
    with open(file, "r", encoding="utf8") as readfile:
        label = readfile.read()
    file = '/'.join(file.split('/')[-3:])
    path = file.replace("txt", "jpg")
    url = "/home/longhn/" + path
    url_text = "/home/longhn/" + file
    a = check_img_too_large(url)
    if a is True:
        copy_img = 'mv ' + url + ' ' + url_error
        copy_text = 'mv ' + url_text + ' ' + url_error
        # print(copy)
        os.system(copy_img)
        os.system(copy_text)
        count += 1
    i +=1
print(i,count)