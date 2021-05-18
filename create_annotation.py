import glob
import os
import time
start = time.time()
with open("/home/fdm/Desktop/chungnph/vietocr/Annotation/train.txt", "w", encoding="utf8") as writefile:
    for file in glob.glob("/home/fdm/Desktop/chungnph/vietocr/Annotation/train_data/*/*.txt"):
        with open(file, "r", encoding="utf8") as readfile:
            label = readfile.read()
        file = '/'.join(file.split('/')[-3:])
        path = file.replace("gt.txt", "jpg")
        writefile.write(f"{path}\t{label}\n")
stop = time.time()
train = stop-start

start = time.time()
with open("/home/fdm/Desktop/chungnph/vietocr/Annotation/valid.txt", "w", encoding="utf8") as writefile:
    for file in glob.glob("/home/fdm/Desktop/chungnph/vietocr/Annotation/val_data/*/*.txt"):
        with open(file, "r", encoding="utf8") as readfile:
            label = readfile.read()
        file = '/'.join(file.split('/')[-3:])
        path = file.replace("gt.txt", "jpg")
        writefile.write(f"{path}\t{label}\n")
stop = time.time()
print('train: ', train)
print('val: ', stop-start)