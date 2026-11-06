영상합성 실습과제 실행 안내
설치: python -m pip install opencv-python numpy
실행: python main.py
창 없이 실행: python main.py --no-window

main.py: 수동 마스크로 흰 배경 전경 입력을 준비한 뒤 합성 실행.
compose.py: 전경 입력을 cvtColor로 회색조 변환하고 threshold로 마스크 생성.
prepare_sticker.py: 수동 마스크로 복잡한 원본 배경을 흰색으로 교체하는 전처리.
edit_mask.py: 왼쪽 지우기/오른쪽 복원/S 저장/Esc 취소.
편집 후 main.py를 다시 실행하면 모든 결과가 갱신됨.

원본 인물 사진에서 윤곽을 자동 추출한 것은 아님.
전처리 마스크 person_mask.png는 수동으로 제작/수정함.
합성 마스크는 sticker_white.png의 회색조 밝기를 임계값 250으로 이진화해 생성함.

배경 출처: https://brunch.co.kr/@ragony/479

