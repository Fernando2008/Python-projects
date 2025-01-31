'''
    INTRODUCTION COMMENT
'''

import os
import cv2
import numpy as np
from keras.models import load_model  # type: ignore
from keras import backend as K

WIDTH, HEIGHT, SIZE = 640, 480, 28
CONFIDENCE_THRESHOLD = 0.9
FRAME_SKIP = 5
PADDING = 15

print("Starting webcam...")
cp = cv2.VideoCapture(0)
cp.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cp.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

if K.image_data_format() == 'channels_first':
    input_shape = (1, SIZE, SIZE)
    first_dim, second_dim = 0, 1
else:
    input_shape = (SIZE, SIZE, 1)
    first_dim, second_dim = 0, 3

def annotate(frame, label, location=(20, 30)):
    cv2.putText(frame, label, location, cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

def extract_digit(frame, rect, pad=PADDING):
    x, y, w, h = rect
    cropped_digit = final_img[y-pad:y+h+pad, x-pad:x+w+pad] / 255.0
    if cropped_digit.shape[0] >= 32 and cropped_digit.shape[1] >= 32:
        return cv2.resize(cropped_digit, (SIZE, SIZE))
    return None

def img_to_mnist(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    return cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, blockSize=321, C=SIZE)

print("Loading model...")
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
model = load_model("mnist_cnn_model.h5")

labelz = dict(enumerate(["zero", "one", "two", "three", "four",
                         "five", "six", "seven", "eight", "nine"]))

frame_count = 0
while True:
    ret, frame = cp.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    final_img = img_to_mnist(frame)
    image_shown = frame.copy()

    contours, _ = cv2.findContours(final_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50]  # Filtra rumore

    for rect in rects:
        x, y, w, h = rect
        mnist_frame = extract_digit(frame, rect)

        if mnist_frame is not None:
            mnist_frame = np.expand_dims(mnist_frame, first_dim)
            mnist_frame = np.expand_dims(mnist_frame, second_dim)

            predictions = model.predict(mnist_frame, verbose=False)
            class_prediction = np.argmax(predictions, axis=-1)[0]
            confidence = np.max(predictions)

            if confidence >= CONFIDENCE_THRESHOLD:
                label = f"{labelz[class_prediction]} ({confidence:.2f})"
                annotate(image_shown, label, location=(x, y - 10))
                cv2.rectangle(image_shown, (x - PADDING, y - PADDING),
                              (x + w + PADDING, y + h + PADDING), color=(0, 255, 0), thickness=2)

    cv2.imshow('frame', image_shown)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cp.release()
cv2.destroyAllWindows()
