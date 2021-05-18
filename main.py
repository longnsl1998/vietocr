#coding: utf-8
import uvicorn
import os
from os import path
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from tool.predictor import Predictor
from tool.config import Cfg

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
app = FastAPI()

config = Cfg.load_config_from_file("config/vgg_transformer.yml")
config['weights'] = f'{os.getenv("MODEL_PATH", "../../models")}/vietocr/transformerocr.pth'
config['cnn']['pretrained'] = False
config['device'] = 'cpu'
detector = Predictor(config)

@app.post('/recognize')
async def recognize(file: UploadFile = File(...)):
    result = detector.predict_bytes(await file.read())
    return {'result': result}

if __name__ == "__main__":
    import sys
    sys.path.append(path.join(path.dirname(__file__), '..'))
    uvicorn.run(app, host="0.0.0.0", port=8002)
