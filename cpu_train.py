from tool.config import Cfg
from model.trainer import Trainer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(model_name, checkpoint_name,url):
    config = Cfg.load_config_from_name(model_name)
    # url = '/home/longhn/Anotation_0106'
    dataset_params = {
        'name':'hw',
        'data_root':'/home/longhn',
        # 'train_root': '/home/fdm/Desktop/chungnph/vietocr/Annotation_2505',
        # 'val_root': '/home/fdm/Desktop/chungnph/vietocr/Annotation_2505',
        'train_annotation': f'{url}/train.txt',
        'valid_annotation': f'{url}/valid.txt'
    }

    params = {
             'print_every':200,
             'valid_every':10*200,
              'iters':100000,
              'checkpoint': f'checkpoint/{checkpoint_name}.pth',
              'export': f'checkpoint/{checkpoint_name}.pth',
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
    config['vocab'] = '''aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐÐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&'()*+,-–./:;<=>?@[\]^_`{|}~ '''
    # config['weights'] = 'checkpoint/ngaycap_0204.pth'
    print(config)
    trainer = Trainer(config, pretrained=True)
    trainer.config.save(f'train_config/{checkpoint_name}.yml')
    trainer.train()

#  main (file config, ten model)
main('vgg_seq2seq','seq2seq_0806_pretrain_32_8k','/home/longhn/Anotation_0806')

# Address:
# iter: 1000000 - valid loss: 0.721 - acc full seq: 0.8174 - acc per char: 0.9423


