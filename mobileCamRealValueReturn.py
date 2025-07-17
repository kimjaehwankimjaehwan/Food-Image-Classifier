# --------------------[라이브러리 로드]--------------------
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import json
import cv2
import time
import threading
from pathlib import Path

# 예외 처리와 함께 라이브러리 로드
try:
    from ultralytics import YOLO
except ImportError:
    st.error("YOLO 라이브러리를 설치해주세요: pip install ultralytics")
    st.stop()

# --------------------[음식 클래스 정의]--------------------
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

# --------------------[데이터베이스 정의]--------------------
FOOD_DATABASE = {
    "백향과": {
        "당뇨병": "good",
        "음식정보": {
            "칼로리": 97,
            "설명": "비타민 C가 풍부하고 당지수가 낮은 열대 과일입니다."
        }
    },
    "베이글샌드위치": {
        "당뇨병": "bad",
        "음식정보": {
            "칼로리": 350,
            "설명": "정제된 탄수화물과 고지방 재료가 많은 샌드위치입니다."
        }
    },
    "보쌈": {
        "당뇨병": "good",
        "음식정보": {
            "칼로리": 550,
            "설명": "삶은 돼지고기로 비교적 기름기가 적고 채소와 함께 섭취 가능합니다."
        }
    },
    "복숭아": {
        "당뇨병": "good",
        "음식정보": {
            "칼로리": 60,
            "설명": "당분은 있지만 수분과 식이섬유가 풍부한 과일입니다."
        }
    },
    "볶음면": {
        "당뇨병": "bad",
        "음식정보": {
            "칼로리": 700,
            "설명": "기름에 볶아 고칼로리이며 혈당을 빠르게 올릴 수 있습니다."
        }
    },
    "볶음밥": {
        "당뇨병": "bad",
        "음식정보": {
            "칼로리": 700,
            "설명": "기름과 탄수화물 비율이 높아 당뇨병에 좋지 않습니다."
        }
    },
    "부침개": {
        "당뇨병": "bad",
        "음식정보": {
            "칼로리": 500,
            "설명": "밀가루와 기름이 많이 들어가 당 지수가 높습니다."
        }
    },
    "비빔밥": {
        "당뇨병": "good",
        "음식정보": {
            "칼로리": 550,
            "설명": "채소와 단백질이 포함되어 있어 균형 잡힌 식사입니다."
        }
    },
    "빵": {
        "당뇨병": "bad",
        "음식정보": {
            "칼로리": 250,
            "설명": "정제된 밀가루로 만들어져 혈당을 빠르게 올립니다."
        }
    },
    "사과파이": {
        "당뇨병": "bad",
        "음식정보": {
            "칼로리": 300,
            "설명": "설탕과 버터가 많이 들어간 디저트입니다."
        }
    }
}

# --------------------[안전한 모델 로드]--------------------
@st.cache_resource
def load_model():
    """YOLO 모델을 안전하게 로드"""
    try:
        model_path = Path("pt/food.pt")
        if not model_path.exists():
            st.warning(f"모델 파일을 찾을 수 없습니다: {model_path}")
            st.info("모델이 없어도 테스트를 위해 임시 모델을 사용합니다.")
            # 임시로 사전 학습된 모델 사용 (테스트용)
            return YOLO('yolov8x.pt')
        return YOLO(str(model_path))
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None

# 모델 로드
model = load_model()
print("✅ 모델 클래스 라벨 확인:", model.names)
if model is None:
    st.stop()

# --------------------[음식 정보 분석 함수]--------------------
def analyze_food_info(class_id, confidence):
    """클래스 ID를 기반으로 음식 정보 분석"""
    try:
        # 클래스 ID를 음식명으로 변환
        if class_id in FOOD_CLASSES:
            food_name = FOOD_CLASSES[class_id]
        else:
            # YOLO 모델의 기본 클래스명 사용 (테스트용)
            food_name = model.names.get(class_id, f"Unknown_{class_id}")
        
        print(f"🔍 탐지된 음식: {food_name} (클래스 ID: {class_id})")  # 디버깅용
        
        # 데이터베이스에서 정보 조회 (안전한 접근)
        if food_name in FOOD_DATABASE:
            food_data = FOOD_DATABASE[food_name]
            diabetes_effect = food_data.get("당뇨병", "unknown")
            food_info = food_data.get("음식정보", {})
            
            effect_text = "당뇨병에 좋습니다 ✅" if diabetes_effect == "good" else "당뇨병에 좋지 않습니다 ⚠️"
            
            return {
                "name": food_name,
                "confidence": confidence,
                "diabetes_effect": effect_text,
                "calorie": f"{food_info.get('칼로리', '정보없음')} kcal",
                "description": food_info.get("설명", "설명이 없습니다."),
                "color": (0, 255, 0) if diabetes_effect == "good" else (0, 165, 255)  # 초록색 or 주황색
            }
        else:
            print(f"⚠️ 데이터베이스에 없는 음식: {food_name}")
            return {
                "name": food_name,
                "confidence": confidence,
                "diabetes_effect": "정보 없음",
                "calorie": "정보 없음",
                "description": f"'{food_name}'은(는) 데이터베이스에 등록되지 않은 음식입니다.",
                "color": (128, 128, 128)  # 회색
            }
    except Exception as e:
        print(f"음식 정보 분석 오류: {e}")
        print(f"오류 발생 위치 - class_id: {class_id}, confidence: {confidence}")
        return {
            "name": f"오류_{class_id}",
            "confidence": confidence,
            "diabetes_effect": "분석 실패",
            "calorie": "정보 없음",
            "description": f"분석 중 오류가 발생했습니다: {str(e)}",
            "color": (0, 0, 255)  # 빨간색
        }

# --------------------[탐지 결과 저장 클래스]--------------------
class DetectionHolder:
    def __init__(self):
        self.current_detection = None
        self.detection_history = []
        self.last_update_time = 0
        self.lock = threading.Lock()
        self.stable_detection_count = 0
        self.min_stable_frames = 3  # 안정적인 탐지를 위한 최소 프레임 수
    
    def update_detection(self, detection_info):
        """탐지 결과를 안정적으로 업데이트"""
        with self.lock:
            current_time = time.time()
            
            # 같은 음식이 연속으로 탐지되는지 확인
            if (self.current_detection and 
                self.current_detection.get('name') == detection_info.get('name')):
                self.stable_detection_count += 1
            else:
                self.stable_detection_count = 1
            
            # 안정적인 탐지일 때만 업데이트
            if self.stable_detection_count >= self.min_stable_frames:
                self.current_detection = detection_info
                self.last_update_time = current_time
                
                # 히스토리 업데이트
                self.detection_history.append((current_time, detection_info['name']))
                if len(self.detection_history) > 20:
                    self.detection_history.pop(0)
    
    def get_current_detection(self):
        """현재 탐지 결과 반환"""
        with self.lock:
            return self.current_detection.copy() if self.current_detection else None
    
    def get_detection_stats(self):
        """탐지 통계 반환"""
        with self.lock:
            if not self.detection_history:
                return {}
            
            # 최근 10개 탐지 결과 통계
            recent_detections = [item[1] for item in self.detection_history[-10:]]
            stats = {}
            for food in recent_detections:
                stats[food] = stats.get(food, 0) + 1
            
            return dict(sorted(stats.items(), key=lambda x: x[1], reverse=True))

holder = DetectionHolder()

# --------------------[개선된 YOLO 영상 처리 클래스]--------------------
class FoodDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.frame_count = 0
        self.process_interval = 2  # 2프레임마다 처리 (성능 최적화)
        self.conf_threshold = 0.3  # 신뢰도 임계값 낮춤
        self.iou_threshold = 0.5
    
    def recv(self, frame):
        try:
            print("📸 프레임 수신됨")
            img = frame.to_ndarray(format="bgr24")
            img = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
            cv2.imwrite("D:/kdt/frame_debug.jpg", img)
            ...
        except Exception as e:
            print(f"❌ recv() 오류: {e}")
        

        # # 이미지 전처리 (품질 향상)
        # img = self.preprocess_image(img)
        
        # 주기적으로만 YOLO 실행
        if self.frame_count % self.process_interval == 0:
            try:
                # YOLO 예측 실행
                results = model.predict(
                    source=img, 
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                    imgsz=640  # 입력 이미지 크기 명시
                )
                
                if results and len(results) > 0:
                    result = results[0]
                    
                    # 탐지 결과 처리
                    self.process_detections(result, img)
                    
                    # 결과 시각화
                    img = self.draw_detections(result, img)
                    
            except Exception as e:
                print(f"YOLO 처리 오류: {e}")
                # 오류 발생시 원본 이미지 반환
                pass
        
        # 현재 탐지 정보를 이미지에 표시
        img = self.draw_current_info(img)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def preprocess_image(self, img):
        """이미지 전처리로 인식률 향상"""
        # 밝기 조정
        img = cv2.convertScaleAbs(img, alpha=1.1, beta=10)
        
        # 노이즈 제거 (선택적)
        # img = cv2.bilateralFilter(img, 9, 75, 75)
        
        return img
    
    def process_detections(self, result, img):
        """탐지 결과 처리"""
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            class_ids = boxes.cls.int().tolist()
            confidences = boxes.conf.tolist()
            
            # 가장 높은 신뢰도의 탐지 결과 선택
            if confidences:
                best_idx = confidences.index(max(confidences))
                best_class_id = class_ids[best_idx]
                best_confidence = confidences[best_idx]
                
                # 음식 정보 분석
                food_info = analyze_food_info(best_class_id, best_confidence)
                if food_info:
                    holder.update_detection(food_info)
    
    def draw_detections(self, result, img):
        """탐지 결과를 이미지에 그리기"""
        try:
            # YOLO 기본 시각화 사용
            annotated_img = result.plot(
                conf=True,  # 신뢰도 표시
                labels=True,  # 라벨 표시
                boxes=True,  # 박스 표시
                line_width=2
            )
            return annotated_img
        except:
            return img
    
    def draw_current_info(self, img):
        """현재 탐지 정보를 이미지에 표시"""
        detection = holder.get_current_detection()
        if detection:
            # 반투명 정보 패널
            overlay = img.copy()
            panel_height = 120
            cv2.rectangle(overlay, (10, 10), (img.shape[1]-10, panel_height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
            
            # 정보 텍스트
            color = detection['color']
            lines = [
                f"음식: {detection['name']} ({detection['confidence']:.1%})",
                f"당뇨병: {detection['diabetes_effect']}",
                f"칼로리: {detection['calorie']}",
                f"업데이트: {time.strftime('%H:%M:%S')}"
            ]
            
            for i, line in enumerate(lines):
                y_pos = 30 + i * 22
                cv2.putText(img, line, (15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        
        return img

# --------------------[Streamlit UI]--------------------
st.set_page_config(page_title="당뇨병 음식 AI", layout="wide")

# 헤더
st.title("🍽️ 당뇨병 음식 판별 AI")
st.markdown("**10가지 음식이 당뇨병에 미치는 영향을 실시간으로 알려드립니다**")

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 카메라 설정
    camera_source = st.selectbox(
        "📷 카메라 선택", 
        ["후면 카메라 (환경)", "전면 카메라 (사용자)"], 
        index=0
    )
    
    # 감지 설정
    st.subheader("🎯 탐지 설정")
    conf_threshold = st.slider("신뢰도 임계값", 0.1, 0.9, 0.3, 0.1)
    
    # 지원 음식 목록
    st.subheader("🍎 지원 음식 (10가지)")
    for i, food in FOOD_CLASSES.items():
        effect = FOOD_DATABASE[food]["당뇨병"]
        emoji = "✅" if effect == "good" else "⚠️"
        st.write(f"{emoji} {food}")
    
    # 탐지 통계
    st.subheader("📊 탐지 통계")
    stats = holder.get_detection_stats()
    if stats:
        for food, count in list(stats.items())[:5]:
            st.write(f"• {food}: {count}회")
    else:
        st.write("아직 탐지된 음식이 없습니다.")

# 메인 영역
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📹 실시간 카메라")
    
    # 카메라 시작/정지
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False
    
    if not st.session_state.camera_active:
        if st.button("🎬 카메라 시작", type="primary", use_container_width=True):
            st.session_state.camera_active = True
            st.rerun()
    else:
        if st.button("⏹️ 카메라 정지", type="secondary", use_container_width=True):
            st.session_state.camera_active = False
            st.rerun()

with col2:
    st.subheader("📋 음식 정보")
    info_placeholder = st.empty()

# 카메라 스트리밍
if st.session_state.camera_active:
    facing_mode = "environment" if "후면" in camera_source else "user"
    
    # WebRTC 스트리머
    webrtc_ctx = webrtc_streamer(
        key="food-detection",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480}
            },
            "audio": False
        },
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        async_processing=True,
        video_processor_factory=FoodDetectionProcessor
    )

    # 상태 표시
    st.write("📡 WebRTC 상태:", webrtc_ctx.state)
    
    # 실시간 정보 업데이트
    while webrtc_ctx.state.playing:
        detection = holder.get_current_detection()
        
        with info_placeholder.container():
            if detection:
                # 당뇨병 영향에 따른 색상
                if "좋습니다" in detection['diabetes_effect']:
                    st.success(f"**{detection['name']}**")
                    st.success(detection['diabetes_effect'])
                else:
                    st.error(f"**{detection['name']}**")
                    st.warning(detection['diabetes_effect'])
                
                # 상세 정보
                st.info(f"**신뢰도:** {detection['confidence']:.1%}")
                st.info(f"**칼로리:** {detection['calorie']}")
                st.write(f"**설명:** {detection['description']}")
                
                # 마지막 업데이트 시간
                update_time = time.time() - holder.last_update_time
                st.caption(f"마지막 업데이트: {update_time:.1f}초 전")
                
            else:
                st.info("🔍 음식을 카메라에 비춰주세요")
                st.write("지원되는 10가지 음식 중 하나를 화면에 보여주세요.")
        
        time.sleep(1)  # 1초마다 UI 업데이트

# 사용 가이드
with st.expander("📖 사용 가이드"):
    st.markdown("""
    ### 🎯 효과적인 음식 인식을 위한 팁
    
    1. **조명**: 밝은 곳에서 촬영하세요
    2. **거리**: 음식과 30-50cm 거리 유지
    3. **각도**: 음식을 정면에서 촬영
    4. **배경**: 단순한 배경 (흰색 접시 권장)
    5. **안정성**: 카메라를 흔들지 말고 고정
    
    ### 🍽️ 지원 음식 목록
    **당뇨병에 좋은 음식 (5가지)**
    - 백향과, 보쌈, 복숭아, 비빔밥
    
    **당뇨병에 좋지 않은 음식 (5가지)**  
    - 베이글샌드위치, 볶음면, 볶음밥, 부침개, 빵, 사과파이
    """)

# 디버깅 정보 (개발용)
if st.checkbox("🔧 디버깅 정보 표시"):
    st.write("**모델 정보:**")
    if model:
        st.write(f"- 모델 클래스 수: {len(model.names) if hasattr(model, 'names') else 'Unknown'}")
        if hasattr(model, 'names'):
            st.write(f"- 모델 클래스: {list(model.names.values())}")
    
    st.write("**현재 탐지 상태:**")
    current_detection = holder.get_current_detection()
    if current_detection:
        st.json(current_detection)
    else:
        st.write("탐지된 음식 없음")

# 푸터
st.markdown("---")
st.markdown("🤖 **당뇨병 음식 판별 AI** | 건강한 식습관을 위한 AI 도우미")