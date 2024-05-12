import cv2
import numpy as np
import pandas as pd
import tkinter as tk
import keras
import json
from PIL import Image
from io import BytesIO
import os

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
    # argmax = np.argmax(pred)
    # label = list(labels_map.keys())[argmax]
    # confidence = pred[argmax]
    return dct, image


def draw_kana():
    root = tk.Tk()
    root.title("Draw Kana")
    root.resizable(True, True)

    def paint(event):
        # print("paint")
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)

    def hint():
        # print("hint")
        path = np.random.choice(dataset["path"])
        item = dataset[dataset["path"] == path].iloc[0]
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (200, 200))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image[image == 0] = 240
        image[image < 245] = 245
        bytes_image = BytesIO()
        image = Image.fromarray(image)
        # change black to gray
        image.save(bytes_image, format="PNG")
        canvas.image = None
        canvas.image = tk.PhotoImage("data", data=bytes_image.getvalue())
        canvas.create_image(0, 0, image=canvas.image, anchor="nw")
        entry.delete(0, "end")
        entry.insert(0, item["kana"])

    def clear():
        # print("clear")
        canvas.delete("all")

    def predict():
        # print("predict")
        ps = canvas.postscript(colormode="color")
        with open("temp.ps", "w") as f:
            f.write(ps)
        img = Image.open("temp.ps")
        img = np.array(img)
        os.remove("temp.ps")
        prediction, _ = predict_image(img)
        text = pretty_predictions(prediction, entry.get() or None)
        guess_label.config(text=text)
        # print(pretty_predictions(prediction))

    def close():
        # print("close")
        root.destroy()

    canvas = tk.Canvas(root, width=200, height=200, bg="white")
    canvas.bind("<B1-Motion>", paint)
    canvas.pack()

    # input box for kana from dataset
    should_be_input = tk.StringVar()
    entry = tk.Entry(root, textvariable=should_be_input)
    entry.pack()

    button = tk.Button(root, text="Predict", command=predict)
    button.pack()

    button = tk.Button(root, text="Clear", command=clear)
    button.pack()

    button = tk.Button(root, text="Set Hint", command=hint)
    button.pack()

    button = tk.Button(root, text="Close", command=close)
    button.pack()

    guess_label = tk.Label(root, text="Draw a kana")
    guess_label.pack()

    root.mainloop()


if __name__ == "__main__":
    draw_kana()
