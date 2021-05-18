FROM python:3.7
RUN apt-get update
RUN pip install --upgrade pip
RUN pip install torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install fastapi uvicorn python-multipart \
    einops gdown PyYAML

ADD ./ /app
WORKDIR /app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]