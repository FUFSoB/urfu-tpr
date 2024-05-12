import pyjson5
import json
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import os
from functools import lru_cache
import pandas as pd
import numpy as np
import hashlib
import cv2

from utils import resize_image


data = pyjson5.load(open("symbols.jsonc"))
hira = kata = str
kana: dict[str : tuple[hira, kata]] = data["kana"]

current_dir = Path(__file__).parent
fonts_dir = current_dir / "fonts"
output_dir = current_dir / "dataset"

fonts_raw = os.popen("fc-list").read().strip().split("\n")
fonts = {font.split(":", 1)[0]: font.split(":", 1)[1] for font in fonts_raw}


def get_font_from_name(name: str):
    for key, value in fonts.items():
        if name in value:
            return key
    return None


@lru_cache
def get_font_from_path(path: Path, size: int):
    font = ImageFont.truetype(path, size)
    return font


def generate_image(
    text: str,
    size=8,
    antialias=True,
    font_path: Path = get_font_from_name("Noto Sans CJK JP"),
    output_dir=output_dir,
    name: str = None,
):
    text_lines = text.split("\n")
    image_size = (size * max(map(len, text_lines)), size * len(text_lines))

    fontmode = "L" if antialias else "1"
    font_name = font_path.stem.replace(" ", "_")
    output_dir_size = output_dir / font_name / str(size)
    output_dir_size.mkdir(exist_ok=True, parents=True)
    file_path = output_dir_size / f"{name or text}_{image_size[0]}_{fontmode}.png"
    if file_path.exists():
        return file_path

    img = Image.new("RGB", image_size, color="white")
    draw = ImageDraw.Draw(img, "RGB")
    draw.fontmode = fontmode

    try:
        font = get_font_from_path(font_path, size - 2)
    except OSError:
        return None

    try:
        draw.text(
            (image_size[0] // 2, image_size[1] // 2 - 1),
            text,
            font=font,
            fill="black",
            anchor="mm",
            align="center",
            spacing=0,
            language="ja",
        )
    except OSError:
        return None
    # if image is still all white, skip
    if img.getextrema() == (255, 255):
        return None

    image = resize_image(np.array(img), size, antialias)
    cv2.imwrite(str(file_path), image)

    return file_path


def generate_json_neural_mapping():
    mapping = {}
    num = 0

    for h, k in kana.values():
        mapping[h] = num
        num += 1
        if h == k:
            continue
        mapping[k] = num
        num += 1

    dumped = json.dumps(mapping, indent=2, ensure_ascii=False)
    output_dir.mkdir(exist_ok=True)
    with open(output_dir / "mapping.json", "w") as f:
        f.write(dumped)


def check_charlist(font: Path):
    data = os.popen(f"./fontcharlist '{font}'").read().strip()
    if "あ" not in data or "ア" not in data:
        return False
    return True


def generate_images():
    # new data table
    data = pd.DataFrame(columns=["path", "kana", "font", "size", "antialias"])
    fonts = list(fonts_dir.iterdir())
    for num, font in enumerate(fonts, 1):
        if not check_charlist(font):
            print("Skipped:", font, "[missing characters]")
            font.unlink()
            continue
        print(font, num, "/", len(fonts))
        skipped = 0
        for h, k in kana.values():
            if skipped >= 2:
                break
            skipped = 0
            for sz in (64,):
                for antialias in (True,):
                    file_path = generate_image(h, sz, antialias, font)
                    if file_path is None:
                        skipped += 1
                        print("Skipped:", font, "[no glyph]", h, font, sz, antialias)
                    else:
                        data = data._append(
                            {
                                "path": file_path,
                                "kana": h,
                                "font": font,
                                "size": sz,
                                "antialias": antialias,
                            },
                            ignore_index=True,
                        )
                    file_path = generate_image(k, sz, antialias, font)
                    if file_path is None:
                        skipped += 1
                        print("Skipped:", font, "[no glyph]", k, font, sz, antialias)
                        continue
                    data = data._append(
                        {
                            "path": file_path,
                            "kana": k,
                            "font": font,
                            "size": sz,
                            "antialias": antialias,
                        },
                        ignore_index=True,
                    )

    data.to_csv(output_dir / "data.csv", index=False)


def generate_previews():
    preview_hiragana = "あいうえお\nかきくけこ\nさしすせそ\nたちつてと\nなにぬねの\nはひふへほ\nまみむめも\nやゆよ\nらりるれろ\nわを\nん"
    preview_katakana = "アイウエオ\nカキクケコ\nサシスセソ\nタチツテト\nナニヌネノ\nハヒフヘホ\nマミムメモ\nヤユヨ\nラリルレロ\nワヲ\nン"
    for font in fonts_dir.iterdir():
        out = current_dir / "previews"
        generate_image(preview_hiragana, 64, True, font, out, "hiragana")
        generate_image(preview_katakana, 64, True, font, out, "katakana")


def checksum_dataset_images():
    dataset = pd.read_csv("dataset/data.csv")
    sums: dict[str, list[str]] = {}
    for path in dataset["path"]:
        try:
            with open(path, "rb") as f:
                data = f.read()
                checksum = hashlib.md5(data).hexdigest()
                if checksum not in sums:
                    sums[checksum] = []
                sums[checksum].append(path)
        except FileNotFoundError:
            dataset = dataset[dataset["path"] != path]

    for checksum, paths in sums.items():
        if len(paths) > 1:
            print(checksum, paths[0], len(paths))
            for path in paths[(0 if len(paths) > 50 else 1) :]:
                print(path)
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                else:
                    dataset = dataset[dataset["path"] != path]

    dataset.to_csv("dataset/data.csv", index=False)


def collage_all_images():
    dataset = pd.read_csv("dataset/data.csv")
    images = [item.path for item in dataset.itertuples() if item.kana == "あ"]
    width = 64
    height = 64
    square = int(len(images) ** 0.5 + 1)
    print(square)
    collage = Image.new("L", (width * square, height * square))
    for i, image_path in enumerate(images):
        x = i % square
        y = i // square
        collage.paste(Image.open(image_path), (x * width, y * height))
    collage.save("collage.png")


if __name__ == "__main__":
    # generate_previews()
    generate_images()
    generate_json_neural_mapping()
    checksum_dataset_images()
    collage_all_images()
