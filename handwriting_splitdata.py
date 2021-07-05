import glob
import numpy as np
import os
import time

start = time.time()
test = open("/home/longhn/handwriting_0307/test.txt", "w", encoding="utf8")
train = open("/home/longhn/handwriting_0307/train.txt", "w", encoding="utf8")
valid = open("/home/longhn/handwriting_0307/valid.txt", "w", encoding="utf8")

all_files = []
for ext in ["*.png", "*.jpeg", "*.jpg"]:
    images = glob.glob(os.path.join("/home/longhn/handwriting_0307/data", ext))
    all_files += images
num_img = len(all_files)
print("total images: %d " % num_img)
num_val = int(num_img * 0.1)
rand_idx = np.random.randint(0, len(all_files), num_val)

url_train = '/home/longhn/handwriting_0307/train/'
url_test = '/home/longhn/handwriting_0307/test/'
dir_train = 'mkdir ' + url_train
dir_test = 'mkdir ' + url_test
if os.path.exists(url_train) is False:
    print('OK')
    os.system(dir_train)
if os.path.exists(url_test) is False:
    os.system(dir_test)
i = 0
for file in glob.glob("/home/longhn/handwriting_0307/data/*.txt"):
    with open(file, "r", encoding="utf8") as readfile:
        label = readfile.read()
    file = '/'.join(file.split('/')[-3:])
    print(file)
    path = file.replace("txt", "jpg")
    url = "/home/longhn/" + path
    url_text = "/home/longhn/" + file
    if not i in rand_idx:
        copy_img = 'cp ' + url + ' ' + url_train
        copy_text = 'cp ' + url_text + ' ' + url_train
        os.system(copy_img)
        os.system(copy_text)
        i += 1
    else:
        copy_img = 'cp ' + url + ' ' + url_test
        copy_text = 'cp ' + url_text + ' ' + url_test
        os.system(copy_img)
        os.system(copy_text)
        i += 1
stop = time.time()
train = stop - start
print('train: ', train)
print("total images train: %d " % num_img)
print("total images test: %d " % num_val)
