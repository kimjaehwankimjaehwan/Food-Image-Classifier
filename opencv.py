import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from ultralytics import YOLO

# ---------- 음식 클래스 ----------
FOOD_CLASSES = {
    0: "백향과",
    1: "베이글샌드위치", 
    2: "보쌈",
    3: "복숭아",
    4: "볶음면",
    5: "볶음밥",
    6: "부침개",
    7: "비빔밥",
    8: "빵",
    9: "사과파이"
}

# ---------- 음식 정보 DB ----------
FOOD_DATABASE = {
    "백향과": {"당뇨병": "good", "칼로리": 97, "설명": "비타민 C 풍부, 당지수 낮음"},
    "베이글샌드위치": {"당뇨병": "bad", "칼로리": 350, "설명": "정제 탄수화물과 고지방"},
    "보쌈": {"당뇨병": "good", "칼로리": 550, "설명": "삶은 돼지고기, 채소와 함께"},
    "복숭아": {"당뇨병": "good", "칼로리": 60, "설명": "수분·식이섬유 풍부"},
    "볶음면": {"당뇨병": "bad", "칼로리": 700, "설명": "기름 많고 혈당 빠르게 상승"},
    "볶음밥": {"당뇨병": "bad", "칼로리": 700, "설명": "기름+탄수화물 조합"},
    "부침개": {"당뇨병": "bad", "칼로리": 500, "설명": "밀가루·기름 다량"},
    "비빔밥": {"당뇨병": "good", "칼로리": 550, "설명": "채소+단백질 균형"},
    "빵": {"당뇨병": "bad", "칼로리": 250, "설명": "정제 밀가루 기반"},
    "사과파이": {"당뇨병": "bad", "칼로리": 300, "설명": "설탕·버터 많은 디저트"}
}

# ---------- 한글 표시 함수 ----------
def draw_korean_text(img, text_lines, position=(10, 10), font_size=22, color=(0, 255, 0)):
    # OpenCV 이미지를 PIL 이미지로 변환
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    # 시스템 폰트 사용 (Windows 기준)
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)  # 한글 폰트
    except:
        font = ImageFont.load_default()

    x, y = position
    for line in text_lines:
        draw.text((x, y), line, font=font, fill=color)
        y += font_size + 5

    return np.array(img_pil)

# ---------- YOLO 모델 로드 ----------
model = YOLO("pt/food.pt")  # 또는 yolov8n.pt

# ---------- 카메라 열기 ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ 카메라 열기 실패")
    exit()

# ---------- 루프 ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 밝기 보정 제거 (alpha=1.0, beta=0)
    frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=0)

    # YOLO 예측
    results = model.predict(frame, conf=0.3, imgsz=640)
    result = results[0]
    boxes = result.boxes
    annotated_frame = result.plot().copy()

    # 탐지 결과 중 가장 신뢰도 높은 것 선택
    if boxes is not None and len(boxes) > 0:
        class_ids = boxes.cls.int().tolist()
        confidences = boxes.conf.tolist()

        if confidences:
            best_idx = confidences.index(max(confidences))
            best_class_id = class_ids[best_idx]
            best_conf = confidences[best_idx]

            food_name = FOOD_CLASSES.get(best_class_id, f"Unknown_{best_class_id}")
            info = FOOD_DATABASE.get(food_name, {})
            effect = info.get("당뇨병", "정보 없음")
            cal = info.get("칼로리", "??")
            desc = info.get("설명", "설명 없음")

            color = (0, 255, 0) if effect == "good" else (0, 165, 255)

            # 텍스트 준비
            lines = [
                f"음식: {food_name}",
                f"칼로리: {cal} kcal",
                f"당뇨병: {'좋음 ✅' if effect == 'good' else '주의 ⚠️'}"
            ]

            # 텍스트 그리기 (한글 지원)
            annotated_frame = draw_korean_text(annotated_frame, lines, position=(20, 20), font_size=24, color=color)

    # 창 표시
    cv2.imshow("🍽️ 당뇨병 음식 탐지", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 종료
cap.release()
cv2.destroyAllWindows()
