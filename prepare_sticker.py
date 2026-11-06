"""수동으로 다듬은 마스크를 이용해 흰 배경 전경 입력을 준비한다."""
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent


def read_image(relative_path, mode=cv2.IMREAD_COLOR):
    path = BASE_DIR / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"파일이 없습니다: {path}")
    image = cv2.imread(str(path), mode)
    # 일부 Windows 한글 경로에서 실패하면 바이트 읽기로 대체한다.
    if image is None:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), mode)
    if image is None:
        raise ValueError(f"영상 읽기 실패: {path}")
    return image


def write_png(relative_path, image):
    path = BASE_DIR / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"영상 인코딩 실패: {path}")
    encoded.tofile(str(path))
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"파일 생성 실패: {path}")


def prepare():
    person = read_image("images/person.jpg")
    manual_mask = read_image("result/person_mask.png", cv2.IMREAD_GRAYSCALE)
    if manual_mask.shape != person.shape[:2]:
        raise ValueError("인물과 수동 전처리 마스크의 크기가 다릅니다.")
    selected = manual_mask > 127
    if not np.any(selected):
        raise ValueError("수동 전처리 마스크에 인물이 없습니다.")
    # 이 수동 마스크는 전경 사진을 정리하는 전처리에 사용한다.
    sticker = np.full_like(person, 255)
    # 인물의 밝은 화소도 배경과 구분되도록 최대 245로 제한한다.
    sticker[selected] = np.minimum(person[selected], 245)
    write_png("images/sticker_white.png", sticker)
    print("전처리 완료: images/sticker_white.png")
    return sticker


if __name__ == "__main__":
    prepare()
