from tool.translate import build_model, translate, translate_beam_search, process_input, predict
from tool.utils import download_weights
from PIL import Image
import io
import torch

class Predictor():
    def __init__(self, config):

        device = config['device']
        
        model, vocab = build_model(config)
        weights = '/tmp/weights.pth'

        if config['weights'].startswith('http'):
            weights = download_weights(config['weights'])
        else:
            weights = config['weights']
        model.load_state_dict(torch.load(weights, map_location=torch.device(device)))

        self.config = config
        self.model = model
        self.vocab = vocab

    def predict_bytes(self, buffer):
        return self.predict(Image.open(io.BytesIO(buffer)))

    def predict(self, img):
        img = process_input(img, self.config['dataset']['image_height'], 
                self.config['dataset']['image_min_width'], self.config['dataset']['image_max_width'])        
        img = img.to(self.config['device'])

        if self.config['predictor']['beamsearch']:
            sent = translate_beam_search(img, self.model)
            s = sent
        else:
            sents = translate(img, self.model)
            s = translate(img, self.model)[0].tolist()

        s = self.vocab.decode(s)

        return s

