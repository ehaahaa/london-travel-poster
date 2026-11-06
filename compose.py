from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "result"
RESULT_DIR.mkdir(exist_ok=True)


def read_image(relative_path, mode=cv2.IMREAD_COLOR):
    path = BASE_DIR / relative_path

    if not path.is_file():
        raise FileNotFoundError(f"파일이 없습니다: {path}")

    image = cv2.imdecode(
        np.fromfile(str(path), dtype=np.uint8), mode
    )

    if image is None:
        raise ValueError(f"사진을 읽을 수 없습니다: {path}")

    return image


def save_image(filename, image):
    path = RESULT_DIR / filename
    success, encoded = cv2.imencode(".png", image)

    if not success:
        raise RuntimeError(f"저장 실패: {path}")

    encoded.tofile(str(path))

    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"파일 생성 실패: {path}")


# 사진과 수정한 마스크 불러오기
background = read_image("images/background.jpeg")
person = read_image("images/person.jpg")
mask = read_image(
    "result/person_mask.png", cv2.IMREAD_GRAYSCALE
)

if mask.shape != person.shape[:2]:
    raise ValueError("인물 사진과 마스크 크기가 다릅니다.")

_, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

if cv2.countNonZero(mask) == 0:
    raise ValueError("마스크에 남아 있는 인물이 없습니다.")

# 인물이 있는 부분만 잘라내기
px, py, pw, ph = cv2.boundingRect(mask)
foreground = person[py:py + ph, px:px + pw]
mask_crop = mask[py:py + ph, px:px + pw]

# 인물을 배경에 맞게 축소
bg_h, bg_w = background.shape[:2]
ratio = min(bg_h * 0.45 / ph, bg_w * 0.48 / pw)

new_w = max(1, round(pw * ratio))
new_h = max(1, round(ph * ratio))

foreground = cv2.resize(foreground, (new_w, new_h))
mask_crop = cv2.resize(
    mask_crop, (new_w, new_h),
    interpolation=cv2.INTER_NEAREST
)

# 전경·배경 통과 마스크
_, mask_fg = cv2.threshold(
    mask_crop, 127, 255, cv2.THRESH_BINARY
)
mask_bg = cv2.bitwise_not(mask_fg)

# 오른쪽 아래에 배치
x = max(0, bg_w - new_w - 10)
y = bg_h - new_h

composite = background.copy()
roi = composite[y:y + new_h, x:x + new_w].copy()

# 배경 부분과 인물 부분을 각각 분리
background_part = cv2.bitwise_and(
    roi, roi, mask=mask_bg
)
foreground_part = cv2.bitwise_and(
    foreground, foreground, mask=mask_fg
)

# 합성 후 배경에 대입
combined = cv2.add(background_part, foreground_part)
composite[y:y + new_h, x:x + new_w] = combined

# 보고서용 중간 결과 저장
save_image("mask_fg.png", mask_fg)
save_image("mask_bg.png", mask_bg)
save_image("roi.png", roi)
save_image("background_part.png", background_part)
save_image("foreground_part.png", foreground_part)
save_image("combined_roi.png", combined)
save_image("first_composite.png", composite)

print("수정한 마스크로 합성 완료!")
print("저장 위치:", RESULT_DIR / "first_composite.png")

# 화면 크기에 맞춰 표시
# 밝기·대비 보정
# 대비 1.08배, 밝기 +8, 범위를 벗어나면 0~255로 제한
before = composite.copy()
corrected = np.clip(
    before.astype(np.float32) * 1.08 + 8,
    0, 255
).astype(np.uint8)

save_image("tone_corrected.png", corrected)

# 처리 전후 수치 비교: 문자를 넣기 전에 측정
gray_before = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
gray_after = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)

comparison = (
    "Contrast multiplier: 1.08\n"
    "Brightness offset: +8\n"
    f"Mean brightness before: {gray_before.mean():.2f}\n"
    f"Mean brightness after: {gray_after.mean():.2f}\n"
    f"Brightness std before: {gray_before.std():.2f}\n"
    f"Brightness std after: {gray_after.std():.2f}\n"
)

(RESULT_DIR / "comparison.txt").write_text(
    comparison, encoding="utf-8"
)
print(comparison)

# 여행 포스터 제목
poster = corrected.copy()
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = bg_w / 415

cv2.putText(
    poster, "LONDON", (20, 55),
    font, 1.2 * font_scale,
    (255, 255, 255), 2, cv2.LINE_AA
)

cv2.putText(
    poster, "MY NEXT WISH DESTINATION", (22, 80),
    font, 0.42 * font_scale,
    (255, 255, 255), 1, cv2.LINE_AA
)

# 포스터 테두리
cv2.rectangle(
    poster, (8, 8), (bg_w - 9, bg_h - 9),
    (255, 255, 255), 1
)

save_image("final_poster.png", poster)

# 아래쪽 기존 표시 코드에서 최종 포스터를 보여주도록 설정
composite = poster

scale = min(600 / bg_w, 800 / bg_h, 1.0)
preview = cv2.resize(
    composite,
    (round(bg_w * scale), round(bg_h * scale))
)

cv2.imshow("Composite", preview)
cv2.waitKey(0)
cv2.destroyAllWindows()