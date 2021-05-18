from tool.config import Cfg
from model.trainer import Trainer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def main(model_name, checkpoint_name):
    config = Cfg.load_config_from_file(f'config/{model_name}.yml')
    dataset_params = {
        'name':'hw',
        'data_root':'/home/fdm/Desktop/chungnph/vietocr/Annotation',
        # 'train_root': '/home/fdm/Desktop/chungnph/vietocr/Annotation',
        # 'val_root': '/home/fdm/Desktop/chungnph/vietocr/Annotation',
        'train_annotation':'train.txt',
        'valid_annotation':'valid.txt'
    }

    params = {
             'print_every':200,
             'valid_every':10*200,
              'iters':200000,
              'checkpoint':f'checkpoint/{checkpoint_name}.pth',
              'export':f'checkpoint/{checkpoint_name}.pth',
              'metrics': 15000,
              'batch_size': 32
             }
    dataloader_params = {
        'num_workers' : 1
    }
    # config['pretrain']['cached'] = 'checkpoint/ngaycap_0204.pth'
    config['trainer'].update(params)
    config['dataset'].update(dataset_params)
    config['dataloader'].update(dataloader_params)
    config['device'] = 'cuda'
    config['vocab'] = '''aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐÐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ '''
    # config['weights'] = 'checkpoint/ngaycap_0204.pth'
    print(config)
    trainer = Trainer(config, pretrained=False)
    trainer.config.save('config.yml')
    trainer.train()

#  main (file config, ten model)
main('vgg-seq2seq', 'address_0504')

# Address:
# iter: 1000000 - valid loss: 0.721 - acc full seq: 0.8174 - acc per char: 0.9423


