import argparse
from PIL import Image
import glob
from tool.predictor import Predictor
from model.trainer import Trainer
from tool.config import Cfg
import cv2

# config = Cfg.load_config_from_file('config/vgg_transformer.yml')
config = Cfg.load_config_from_file('config_seq2seq.yml')
dataset_params = {
    'name':'hw',
    'data_root':'/home/fdm/Desktop/chungnph/vietocr/Annotation',
    'train_annotation':'train.txt',
    'valid_annotation':'valid.txt'
}

params = {
         'print_every':200,
         'valid_every':10*200,
          'iters':200000,
          'checkpoint':'checkpoint/Name_1803(add_12kname_140kgenname)_seq2seq.pth',
          'export':'checkpoint/Name_1803(add_12kname_140kgenname)_seq2seq.pth',
          'metrics': 10000000,
          'batch_size': 32
         }


dataloader_params = {
    'num_workers' : 1
}

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['dataloader'].update(dataloader_params)
config['device'] = 'cuda'
config['vocab'] = '''aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐÐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ '''
config['weights'] = '/home/fdm/Desktop/chungnph/vietocr/run/checkpoint/Name_1803(add_12kname_140kgenname)_seq2seq.pth'
print(config)

# detector = Predictor(config)
# img_path = '/home/fdm/Desktop/chungnph/vietocr/Data/NAME/1K_CMT_NAME_OK/Border_CCCD_1CMND_a_Hung_-_scan_resize.pdf20201027175611702020111308332898.jpg'
# img = Image.open(img_path)
# s = detector.predict(img)
# print(s)

trainer = Trainer(config, pretrained=False)
trainer.val()

#seq2seq
# 0.6891206756851377
# 0.9534023668639053 0.9904031

#tran
# 0.6947050598951486
# 0.944896449704142 0.9899969