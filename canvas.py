from PIL import Image, ImageDraw
import numpy as np
import sys
import os
from pathlib import Path
from io import StringIO

from utils import pretty_predictions

current_dir = Path(__file__).parent
output_dir = current_dir / "canvas"


def create_canvas() -> Image.Image:
    # Create a new image with 4x4 grid of 64x64 cells without borders
    img = Image.new("L", (64 * 4 + 1, 64 * 4 + 1), color="white")
    draw = ImageDraw.Draw(img)
    for i in range(4):
        for j in range(4):
            draw.rectangle(
                (i * 64, j * 64, (i + 1) * 64, (j + 1) * 64),
                outline="black",
            )
    return img



def parse_canvas(canvas: Image.Image) -> list[Image.Image]:
    # Parse the canvas into 16 64x64 images
    images = []
    for i in range(4):
        for j in range(4):
            images.append(canvas.crop((i * 64 + 1, j * 64 + 1, (i + 1) * 64, (j + 1) * 64)))
    return images


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    if not arg:
        print("Usage: python canvas.py <create|parse>")
        sys.exit(1)

    if arg == "create":
        canvas = create_canvas()
        canvas.save("canvas.png")
    elif arg == "parse":
        output_dir.mkdir(exist_ok=True)
        canvas = Image.open("canvas.png")
        images = parse_canvas(canvas)
        import model
        import cv2
        buffer = StringIO()
        def print(*args, **kwargs):
            __builtins__.print(*args, **kwargs, file=buffer)
            __builtins__.print(*args, **kwargs)

        for i, image in enumerate(images, 1):
            prediction, img = model.predict_image(np.array(image))
            file_path = output_dir / f"cell_{i}.png"
            cv2.imwrite(str(file_path), img)
            os.system(f"convert {file_path} -geometry 32x32 sixel:-")
            print()
            print(f"Cell {i}:")
            print(pretty_predictions(prediction))
        buffer.seek(0)
        data = buffer.getvalue()
        with open(output_dir / f"guesses.txt", "w") as f:
            f.write(data)
