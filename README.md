# Food-Image-Classifier
A program that tells you whether a food is good or bad for diabetes based on its image
## 안녕하세요.  한국 인공지능 소프트웨어산업협회 KDT 강사교육 2조입니다. 

## 저희 조는 이승준 강사님, 명진숙 강사님, 김문천 강사님, 그리고 저 이렇게 구성되어 있으며 이승준 강사님이 팀장을 맡았습니다.

## 데이터는 AIHUB 의 건강관리를 위한 음식 이미지 데이터를 사용하였으며 500여 종 중에 10종만을 선별하여 사용하였습니다. (https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=242)

## 다음은 코드와 음식의 이름입니다.   
  0: 백향과
  1: 베이글샌드위치
  2: 보쌈
  3: 복숭아
  4: 볶음면
  5: 볶음밥
  6: 부침개
  7: 비빔밥
  8: 빵
  9: 사과파이

## 내부 로컬망에  yolo 모델을 이용하여 캠카메라 혹은 핸드폰 카메라로 음식을 찍으면 객체를 탐지하고 해당 음식이 당뇨병의 영향에 대해 설명하는 서비스를 구현하였습니다.   
