import cv2
import numpy as np

def resize_image(image: np.ndarray, size: int, antialias: bool = False) -> np.ndarray:
    interpolation = cv2.INTER_NEAREST if not antialias else cv2.INTER_LINEAR
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find non-white pixels
    non_white_pixels = cv2.findNonZero(cv2.bitwise_not(image_gray))

    if non_white_pixels is None:
        return np.ones((size, size), dtype=np.uint8) * 255

    # Get the bounding box of non-white pixels
    x, y, w, h = cv2.boundingRect(non_white_pixels)

    # Crop the image to the bounding box
    cropped_image = image_gray[y:y+h, x:x+w]

    # Calculate padding for centering
    pad_x = max((size - w) // 2, 0)
    pad_y = max((size - h) // 2, 0)

    # Calculate the size of the cropped image
    crop_width = min(size, w)
    crop_height = min(size, h)

    # Resize the cropped image to fit the canvas size while maintaining aspect ratio
    new_width = new_height = size
    if crop_width > crop_height:
        new_height = int(size * crop_height / crop_width)
        cropped_image = cv2.resize(cropped_image, (size, new_height), interpolation=interpolation)
    else:
        new_width = int(size * crop_width / crop_height)
        cropped_image = cv2.resize(cropped_image, (new_width, size), interpolation=interpolation)

    # Calculate padding for centering
    pad_x = max((size - new_width) // 2, 0)
    pad_y = max((size - new_height) // 2, 0)

    # Create a canvas of size (size, size)
    canvas = np.ones((size, size), dtype=np.uint8) * 255

    # Place the resized cropped image onto the canvas
    canvas[pad_y:pad_y+cropped_image.shape[0], pad_x:pad_x+cropped_image.shape[1]] = cropped_image

    return canvas


def pretty_predictions(predictions: dict[str, float], should_be: str = None) -> None:
    a = []
    for label, confidence in predictions.items():
        if confidence < 0.01:
            break
        text = f"{label}: {confidence * 100:.2f}%"
        if should_be and label == should_be:
            text = f"{text} <- Correct!"
        a.append(text)

    return "\n".join(a)
