# -*- coding: utf-8 -*-
"""
Created on Fri May 21 09:07:18 2021

@author: fdm
"""
import cv2
import numpy as np
import glob
import ntpath
import os
import re
from PIL import Image
import time
from tool.predictor import Predictor
from tool.config import Cfg
config_all = Cfg.load_config_from_file('./train_config/seq2seq_handwriting_0307_pretrain_32_2k.yml')
config_all['weights'] = './checkpoint/seq2seq_handwriting_0307_pretrain_32_2k.pth'
config_all['cnn']['pretrained'] = False
config_all['device'] = 'cuda:0'
# config_all['device'] = 'cuda:1'
config_all['predictor']['beamsearch'] = False
# config_all['vocab'] = '''aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0125456789!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ '''
detector_old = Predictor(config_all)


def no_accent_vietnamese(s):
	s = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', s)
	s = re.sub(r'[ÀÁẠẢÃĂẰẮẶẲẴÂẦẤẬẨẪ]', 'A', s)
	s = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', s)
	s = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', s)
	s = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', s)
	s = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', s)
	s = re.sub(r'[ìíịỉĩ]', 'i', s)
	s = re.sub(r'[ÌÍỊỈĨ]', 'I', s)
	s = re.sub(r'[ùúụủũưừứựửữ]', 'u', s)
	s = re.sub(r'[ƯỪỨỰỬỮÙÚỤỦŨ]', 'U', s)
	s = re.sub(r'[ỳýỵỷỹ]', 'y', s)
	s = re.sub(r'[ỲÝỴỶỸ]', 'Y', s)
	s = re.sub(r'[Đ]', 'D', s)
	s = re.sub(r'[đ]', 'd', s)
	return s


def util_check_input_img(img_input):
	if not isinstance(img_input, (np.ndarray)):
		#print('not np array')
		return False
	
	if img_input.shape[0] < 10 or img_input.shape[1] < 10:
		#print('small image %d %d' %(img_input.shape[0], img_input.shape[1]))
		return False
	return True


def run_ocr_cnn(img_line):
	if not util_check_input_img(img_line):
		return ""
	is_success, buffer = cv2.imencode('.jpg', img_line)

	# print(buffer.tobytes())
	str_ocr_val = detector_old.predict_bytes(buffer)
	return str_ocr_val	


def Repalce(s):
    last_symbol = s[len(s)-1]
    for j in ['/', '_', ':', ';', '-', '.', ',', '?']:
        if last_symbol == j:
            s = s.replace(j, '')
    for l in ['/', '_', ':', ';', '-', '.', ',', '?']:
        s = s.replace(l, '')
    s = s.replace('  ', ' ')
    return s.strip()


def check_cnn_model(str_img_path, txt_file):
	### read and run ocr by cnn model
	img_in = cv2.imread(str_img_path)
	str_ocr = run_ocr_cnn(img_in)
	kq_test = ''
	if len(str_ocr)==0 or len(txt_file)==0:
		return [0, 0, 0, "false"]
	if Repalce(str_ocr).strip() == Repalce(txt_file).strip():
		a = 1
	else:
		a = 0
		kq_test = f"\n{str_ocr}\n{txt_file}\n-------------------------"
	upperocr = no_accent_vietnamese(Repalce(str_ocr).strip())
	uppertxt = no_accent_vietnamese(Repalce(txt_file).strip())
	if no_accent_vietnamese(Repalce(str_ocr).strip()) == no_accent_vietnamese(Repalce(txt_file).strip()):
		b = 1
	else:
		b = 0
	if upperocr.upper() == uppertxt.upper():
		c = 1
	else:
		c = 0
	return [a, b, c, kq_test]
	# print(a)
	### TO DO: read txt_file
	### str_true = read from text file
	### compare in 2 level:
	### 1. compare str_ocr and str_true	
	### 2. No accent vnese
	## str_non_vnese_ocr = 	no_accent_vietnamese (str_ocr)
	## str_non_vnese_true = no_accent_vietnamese (str_true)
	## compare str_non_vnese_ocr and str_non_vnese_true
	

# check_cnn_model('/home/longhn/Desktop/Anotation/data/0HD_QuanLyTaiKhoan_COLOMBO.pdf_182021042210545568.jpg','CÔNG TY TNHH ĐẦU TƯ VÀ PHÁT TRIỂN COLOMBO')
afile = open('/home/longhn/handwriting_0307/test.txt')
kq = open('/home/longhn/handwriting_0307/kq.txt', 'w', encoding="utf8")
true_11 = 0
true_noacc = 0
true_upper = 0
start = time.time()
for x in afile:
	data = x.split("\t")
	# print(data[0])
	link = '/home/longhn/' + data[0]
	text = data[1].strip()
	# print(link)
	if check_cnn_model(link, text)[3] != '':
		kq.write(f"{link} {check_cnn_model(link,text)[3]}")
		# kq.write(check_cnn_model(link,text)[3])
	true_11 += check_cnn_model(link, text)[0]
	true_noacc += check_cnn_model(link, text)[1]
	true_upper += check_cnn_model(link, text)[2]
stop = time.time()
print("So sanh 1vs1:",true_11)
print("So sanh ko dau:", true_noacc)
print("So sanh upper:", true_upper)
print("Tong thoi gian:", stop-start)
