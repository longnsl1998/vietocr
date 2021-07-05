import glob
import numpy as np
import os
import time


start = time.time()
valid = open("/home/longhn/handwriting_0307/valid.txt", "w", encoding="utf8")
all_files = []
for ext in ["*.png", "*.jpeg", "*.jpg"]:
  images = glob.glob(os.path.join("/home/longhn/handwriting_0307/train", ext))
  all_files += images
num_img = len(all_files)
print("total images: %d " %num_img)
num_val = int(num_img*0.1)
rand_idx = np.random.randint(0, len(all_files), num_val)
with open("/home/longhn/handwriting_0307/train.txt", "w", encoding="utf8") as writefile:
    i = 0
    for file in glob.glob("/home/longhn/handwriting_0307/train/*.txt"):
        with open(file, "r", encoding="utf8") as readfile:
            label = readfile.read()
        file = '/'.join(file.split('/')[-3:])
        print(file)
        path = file.replace("txt", "jpg")
        if not i in rand_idx:
            writefile.write(f"{path}\t{label}\n")
            i += 1
        else:
            valid.write(f"{path}\t{label}\n")
            i += 1


# Creat test file
test = open("/home/longhn/handwriting_0307/test.txt", "w", encoding="utf8")
i = 0
for file in glob.glob("/home/longhn/handwriting_0307/test/*.txt"):
    with open(file, "r", encoding="utf8") as readfile:
        label = readfile.read()
    file = '/'.join(file.split('/')[-3:])
    print(file)
    path = file.replace("txt", "jpg")
    test.write(f"{path}\t{label}\n")

# test = open("/home/longhn/Desktop/result_label_SMD/Anotation_0506/p1/test.txt", "w", encoding="utf8")
# i = 0
# for file in glob.glob("/home/longhn/Desktop/result_label_SMD/Anotation_0506/p1/*.txt"):
#     with open(file, "r", encoding="utf8") as readfile:
#         label = readfile.read()
#     file = '/'.join(file.split('/')[-3:])
#     print(file)
#     path = file.replace("txt", "jpg")
#     test.write(f"{path}\t{label}\n")
stop = time.time()
train = stop-start
print('train: ', train)