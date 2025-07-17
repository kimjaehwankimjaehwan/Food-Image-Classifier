from ultralytics import YOLO

# 1. 모델 로드
model = YOLO("pt/food.pt")  # 또는 yolov8n.pt

# 2. 테스트 이미지 경로
img_path = "test_image.jpg"

# 3. 예측 + 결과 보기
results = model.predict(img_path, show=True)  # show=True → bounding box 띄움
