docker build -t vietocr:latest .

docker run -d --name ekyc-ocr -v D:/Docker/ekyc/vietocr/models:/app/models -e MODEL_PATH=/app/models -p 9000:8000 vietocr:latest