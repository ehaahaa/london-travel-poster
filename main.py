from pathlib import Path
import cv2
import numpy as np

# main.py가 있는 폴더를 기준으로 경로 설정
BASE_DIR = Path(__file__).resolve().parent


def read_image(filename):
    path = BASE_DIR / "images" / filename

    # 파일이 실제로 있는지 확인
    if not path.is_file():
        raise FileNotFoundError(f"사진 파일이 없습니다: {path}")

    # 한글이 포함된 윈도우 경로에서도 읽을 수 있도록 처리
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"사진을 읽을 수 없습니다: {path}")

    return image


def show_image(title, image):
    # 화면에 들어오도록 표시용 사진만 축소
    height, width = image.shape[:2]
    scale = min(600 / width, 750 / height, 1.0)

    preview = cv2.resize(
        image,
        (round(width * scale), round(height * scale))
    )
    cv2.imshow(title, preview)


# 사진 불러오기
background = read_image("background.jpeg")
person = read_image("person.jpg")

# 사진 크기 출력: 너비 × 높이
print("배경 사진:", background.shape[1], "x", background.shape[0])
print("인물 사진:", person.shape[1], "x", person.shape[0])

# 인물 사진을 클릭하기 편한 크기로 축소
h, w = person.shape[:2]
scale = min(700 / w, 800 / h, 1.0)
preview = cv2.resize(
    person, (round(w * scale), round(h * scale))
)

points = []


def click_outline(event, x, y, flags, param):
    # 왼쪽 클릭: 테두리 점 추가
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))

    # 오른쪽 클릭: 마지막 점 취소
    elif event == cv2.EVENT_RBUTTONDOWN and points:
        points.pop()


cv2.namedWindow("Select Person")
cv2.setMouseCallback("Select Person", click_outline)

print("몸의 테두리를 순서대로 왼쪽 클릭하세요.")
print("오른쪽 클릭: 마지막 점 취소")
print("Enter: 선택 완료 / Esc: 취소")

completed = False

while True:
    canvas = preview.copy()

    # 선택한 점과 연결선 표시
    for point in points:
        cv2.circle(canvas, point, 3, (0, 0, 255), -1)

    if len(points) >= 2:
        cv2.polylines(
            canvas,
            [np.array(points, dtype=np.int32)],
            False, (0, 255, 0), 1
        )

    cv2.imshow("Select Person", canvas)
    key = cv2.waitKey(20) & 0xFF

    if key == 13 and len(points) >= 3:
        completed = True
        break
    elif key == 27:
        break

cv2.destroyAllWindows()

if completed:
    # 원본 크기의 검은 마스크 생성
    mask = np.zeros(person.shape[:2], dtype=np.uint8)

    # 클릭한 좌표를 원본 사진 좌표로 변환
    original_points = np.array([
        (
            round(x * w / preview.shape[1]),
            round(y * h / preview.shape[0])
        )
        for x, y in points
    ], dtype=np.int32)

    # 선택한 테두리 안쪽을 흰색으로 채우기
    cv2.fillPoly(mask, [original_points], 255)

    # 흰색 부분의 인물만 남기기
    cutout = cv2.bitwise_and(person, person, mask=mask)

    # 결과 저장 폴더
result_dir = BASE_DIR / "result"
result_dir.mkdir(exist_ok=True)


def save_image(filename, image):
    path = result_dir / filename
    success, encoded = cv2.imencode(".png", image)

    if not success:
        raise RuntimeError(f"저장 실패: {path}")

    encoded.tofile(str(path))


# 원본 크기의 마스크 저장
save_image("person_mask.png", mask)

# 인물이 있는 부분만 사각형으로 잘라내기
px, py, pw, ph = cv2.boundingRect(mask)
foreground = person[py:py + ph, px:px + pw]
cropped_mask = mask[py:py + ph, px:px + pw]

# 배경 높이의 45% 크기로 인물 축소
bg_h, bg_w = background.shape[:2]
ratio = min(
    bg_h * 0.45 / ph,
    bg_w * 0.48 / pw
)
new_w = max(1, round(pw * ratio))
new_h = max(1, round(ph * ratio))

foreground = cv2.resize(foreground, (new_w, new_h))
resized_mask = cv2.resize(
    cropped_mask, (new_w, new_h),
    interpolation=cv2.INTER_NEAREST
)

# 이진 전경 마스크와 반대인 배경 마스크
_, mask_fg = cv2.threshold(
    resized_mask, 127, 255, cv2.THRESH_BINARY
)
mask_bg = cv2.bitwise_not(mask_fg)

# 오른쪽 아래를 합성 위치로 설정
x = max(0, bg_w - new_w - 10)
y = bg_h - new_h

composite = background.copy()

# 인물과 같은 크기의 배경 영역(ROI)
roi = composite[y:y + new_h, x:x + new_w]

# 배경 부분과 인물 부분을 각각 분리
background_part = cv2.bitwise_and(
    roi, roi, mask=mask_bg
)
foreground_part = cv2.bitwise_and(
    foreground, foreground, mask=mask_fg
)

# 두 부분을 더한 뒤 배경에 넣기
combined = cv2.add(background_part, foreground_part)
composite[y:y + new_h, x:x + new_w] = combined

# 중간 결과와 합성 사진 저장
save_image("mask_fg.png", mask_fg)
save_image("mask_bg.png", mask_bg)
save_image("background_part.png", background_part)
save_image("foreground_part.png", foreground_part)
save_image("first_composite.png", composite)

print("result 폴더에 마스크와 합성 결과를 저장했습니다.")

show_image("Composite", composite)
cv2.waitKey(0)
cv2.destroyAllWindows()