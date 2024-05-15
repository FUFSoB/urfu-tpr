import cv2
import numpy as np
import pandas as pd
import tkinter as tk
import keras
import json
from PIL import Image
from io import BytesIO
import time
import threading

from utils import resize_image, pretty_predictions

dataset = pd.read_csv("dataset/data.csv")
size = dataset["size"][0]
antialias = dataset["antialias"][0]
labels_map = json.load(open("dataset/mapping.json"))
model: keras.models.Sequential = keras.models.load_model("output/model.keras")


def predict_image(image: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    image = resize_image(image, size, antialias)
    img = image
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    img = img / 255.0
    pred = model.predict(img)[0]
    dct = {
        list(labels_map.keys())[i]: round(pred[i], 4)
        for i in range(len(labels_map))
    }
    dct = {k: v for k, v in sorted(dct.items(), key=lambda item: item[1], reverse=True)}
    return dct, image


def draw_kana():
    root = tk.Tk()
    root.title("Draw Kana")
    root.resizable(True, True)

    buttons_frame = tk.Frame(root)

    def threaded(fn):
        def wrapper(*args, **kwargs):
            t = threading.Thread(target=fn, args=args, kwargs=kwargs)
            t.start()

        return wrapper

    def paint(event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)

    def hint():
        path = np.random.choice(dataset["path"])
        item = dataset[dataset["path"] == path].iloc[0]
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (200, 200))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image[image == 0] = 240
        image[image < 245] = 245
        bytes_image = BytesIO()
        image = Image.fromarray(image)
        image.save(bytes_image, format="PNG")
        canvas.image = None
        canvas.image = tk.PhotoImage("hint", data=bytes_image.getvalue())
        canvas.create_image(0, 0, image=canvas.image, anchor="nw")
        entry.delete(0, "end")
        entry.insert(0, item["kana"])

    def clear():
        canvas.delete("all")

    def predict():
        ps = canvas.postscript(colormode="color")
        img = Image.open(BytesIO(bytes(ps,'ascii')))
        img = np.array(img)
        prediction, new_image = predict_image(img)
        new_image = Image.fromarray(new_image)
        b = BytesIO()
        new_image.save(b, format="PNG")
        predict_canvas.image = None
        predict_canvas.image = tk.PhotoImage("prediction", data=b.getvalue())
        predict_canvas.create_image(0, 0, image=predict_canvas.image, anchor="nw")
        text = pretty_predictions(prediction, entry.get() or None)
        guess_label.config(text=text)

    def close():
        root.destroy()

    real_time = tk.BooleanVar()

    canvas = tk.Canvas(root, width=200, height=200, bg="white")
    canvas.bind("<B1-Motion>", paint)
    canvas.bind("<ButtonRelease-1>", lambda e: real_time.get() and threaded(predict)())
    canvas.pack()

    should_be_input = tk.StringVar()
    entry = tk.Entry(buttons_frame, textvariable=should_be_input)
    entry.pack()

    button = tk.Button(buttons_frame, text="Predict", command=threaded(predict))
    button.pack()

    checkbox = tk.Checkbutton(buttons_frame, text="Real-time", variable=real_time)
    checkbox.pack()

    button = tk.Button(buttons_frame, text="Clear", command=clear)
    button.pack()

    button = tk.Button(buttons_frame, text="Set Hint", command=hint)
    button.pack()

    button = tk.Button(buttons_frame, text="Close", command=close)
    button.pack()

    predict_canvas = tk.Canvas(buttons_frame, width=64, height=64, bg="white")
    predict_canvas.pack()

    guess_label = tk.Label(buttons_frame, text="Draw a kana")
    guess_label.pack()

    buttons_frame.pack()
    root.mainloop()


if __name__ == "__main__":
    draw_kana()
