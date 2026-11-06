from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent


def load_image(path, mode):
    if not path.is_file():
        raise FileNotFoundError(f"파일이 없습니다: {path}")

    image = cv2.imdecode(
        np.fromfile(str(path), dtype=np.uint8), mode
    )
    if image is None:
        raise ValueError(f"사진을 읽을 수 없습니다: {path}")
    return image


person = load_image(
    BASE_DIR / "images" / "person.jpg",
    cv2.IMREAD_COLOR
)

mask_path = BASE_DIR / "result" / "person_mask.png"
mask = load_image(mask_path, cv2.IMREAD_GRAYSCALE)

if mask.shape != person.shape[:2]:
    raise ValueError("인물 사진과 마스크 크기가 다릅니다.")

# 수정 전 마스크도 보관
backup = mask_path.with_name("person_mask_backup.png")
if not backup.exists():
    success, encoded = cv2.imencode(".png", mask)
    if not success:
        raise RuntimeError("백업 저장 실패")
    encoded.tofile(str(backup))

# 화면에 표시할 크기
h, w = mask.shape
scale = min(750 / w, 850 / h, 1.0)
view_w = round(w * scale)
view_h = round(h * scale)

radius = 5
previous = None


def mouse(event, x, y, flags, param):
    global previous

    # 왼쪽 버튼: 지우기 / 오른쪽 버튼: 복원
    if flags & cv2.EVENT_FLAG_LBUTTON:
        value = 0
    elif flags & cv2.EVENT_FLAG_RBUTTON:
        value = 255
    else:
        previous = None
        return

    # 화면 좌표를 원본 좌표로 변환
    point = (
        min(w - 1, max(0, round(x * w / view_w))),
        min(h - 1, max(0, round(y * h / view_h)))
    )
    brush = max(1, round(radius / scale))

    if previous is not None:
        cv2.line(mask, previous, point, value, brush * 2)

    cv2.circle(mask, point, brush, value, -1)
    previous = point


cv2.namedWindow("Mask Editor")
cv2.setMouseCallback("Mask Editor", mouse)

print("왼쪽 드래그: 지우기 / 오른쪽 드래그: 복원")
print("작은 붓: - / 큰 붓: +")
print("S: 저장 후 종료 / Esc: 저장하지 않고 종료")

while True:
    # 지운 곳은 회색으로 표시
    preview = np.full_like(person, 180)
    preview[mask > 127] = person[mask > 127]
    preview = cv2.resize(preview, (view_w, view_h))

    cv2.putText(
        preview, f"Brush: {radius} | S: Save",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 0, 255), 2
    )
    cv2.imshow("Mask Editor", preview)

    key = cv2.waitKey(20) & 0xFF

    if key in (ord("s"), ord("S")):
        success, encoded = cv2.imencode(".png", mask)
        if not success:
            raise RuntimeError("마스크 저장 실패")
        encoded.tofile(str(mask_path))
        print("수정한 마스크 저장 완료!")
        break
    elif key == 27:
        break
    elif key in (ord("+"), ord("=")):
        radius = min(40, radius + 1)
    elif key == ord("-"):
        radius = max(1, radius - 1)

cv2.destroyAllWindows()