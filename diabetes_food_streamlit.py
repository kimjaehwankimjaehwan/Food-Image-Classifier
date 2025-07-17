import streamlit as st
import cv2
import time
from ultralytics import YOLO

# YOLO 모델 로드 (자신의 학습된 모델 경로로 변경)
model = YOLO("pt/food.pt")

# 클래스명 매핑
FOOD_CLASSES = {
    0: "백향과", 1: "베이글샌드위치", 2: "보쌈", 3: "복숭아", 4: "볶음면",
    5: "볶음밥", 6: "부침개", 7: "비빔밥", 8: "빵", 9: "사과파이"
}

# 당뇨병 관련 DB
FOOD_DATABASE = {
    "백향과": {"당뇨병": "good", "칼로리": 97, "설명": "비타민 C 풍부, 당지수 낮음"},
    "베이글샌드위치": {"당뇨병": "bad", "칼로리": 350, "설명": "정제 탄수화물, 고지방"},
    "보쌈": {"당뇨병": "good", "칼로리": 550, "설명": "삶은 돼지고기, 채소와 함께"},
    "복숭아": {"당뇨병": "good", "칼로리": 60, "설명": "당분은 있지만 식이섬유 풍부"},
    "볶음면": {"당뇨병": "bad", "칼로리": 700, "설명": "기름에 볶아 혈당 상승 유발"},
    "볶음밥": {"당뇨병": "bad", "칼로리": 700, "설명": "기름과 탄수화물 비율 높음"},
    "부침개": {"당뇨병": "bad", "칼로리": 500, "설명": "밀가루와 기름 많음"},
    "비빔밥": {"당뇨병": "good", "칼로리": 550, "설명": "채소와 단백질 균형"},
    "빵": {"당뇨병": "bad", "칼로리": 250, "설명": "정제 밀가루로 혈당 상승"},
    "사과파이": {"당뇨병": "bad", "칼로리": 300, "설명": "설탕과 버터가 많음"}
}

# Streamlit 구성
st.set_page_config(page_title="당뇨병 음식 AI", layout="wide")
st.title("🍽️ 당뇨병 음식 탐지 AI")
st.markdown("YOLOv8로 인식한 음식이 당뇨병에 미치는 영향을 알려줍니다.")

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("❌ 웹캠을 열 수 없습니다.")
    st.stop()

frame_area = st.empty()
info_area = st.empty()

# 상태 유지 변수
last_detected = None
same_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.3, verbose=False)
        result = results[0]

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            class_ids = boxes.cls.int().tolist()
            confidences = boxes.conf.tolist()

            best_idx = confidences.index(max(confidences))
            class_id = class_ids[best_idx]
            food_name = FOOD_CLASSES.get(class_id, "알 수 없음")
            confidence = confidences[best_idx]

            if last_detected == food_name:
                same_count += 1
            else:
                same_count = 1
                last_detected = food_name

            if same_count >= 2:
                food_data = FOOD_DATABASE.get(food_name, {})
                effect = food_data.get("당뇨병", "unknown")
                emoji = "✅" if effect == "good" else "⚠️"
                calorie = food_data.get("칼로리", "N/A")
                desc = food_data.get("설명", "정보 없음")

                info_area.info(f"""
                **🍱 음식**: {food_name}  
                **🧠 신뢰도**: {confidence:.1%}  
                **🩺 당뇨병 영향**: {emoji} {'좋음' if effect=='good' else '나쁨'}  
                **🔥 칼로리**: {calorie} kcal  
                **📘 설명**: {desc}
                """)

        annotated = result.plot()
        frame_area.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
        time.sleep(0.2)

except Exception as e:
    st.error(f"오류 발생: {e}")

finally:
    cap.release()
