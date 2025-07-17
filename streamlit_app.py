## streamlit + OpenCV

import streamlit as st
import cv2
import time
from ultralytics import YOLO

# 모델 로드 (YOLOv8 사전 학습 모델 사용)
model = YOLO("yolov8n.pt")

# Streamlit 설정
st.set_page_config(page_title="YOLO 실시간 탐지", layout="wide")
st.title("🍽️ OpenCV + Streamlit 음식 탐지")

# 웹캠 연결
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("❌ 웹캠을 열 수 없습니다.")
    st.stop()

# 출력용 이미지 영역
frame_slot = st.empty()

# 프레임 루프
try:
    while True:
        success, frame = cap.read()
        if not success:
            st.warning("⚠️ 프레임 수신 실패")
            break

        # YOLO 탐지 실행
        results = model.predict(source=frame, imgsz=640, conf=0.3, verbose=False)
        annotated = results[0].plot()

        # BGR → RGB 변환 후 표시
        frame_slot.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        # 0.1초 대기
        time.sleep(0.1)

except Exception as e:
    st.error(f"에러 발생: {e}")

finally:
    cap.release()
