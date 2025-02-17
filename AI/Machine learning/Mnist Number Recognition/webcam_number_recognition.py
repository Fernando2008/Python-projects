'''
    This script uses a webcam feed to detect and classify handwritten digits using a pre-trained MNIST model.
    The script processes frames from the webcam, extracts potential digit areas, and classifies them 
    only if the model's confidence exceeds a specified threshold.

    author: Fernando Togna
    creation: Junuary 2025
    last update: Junuary 2025
    email: fernandotogna2@gmail.com
'''

import os
import cv2
import numpy as np
from keras.models import load_model  # type: ignore
from keras import backend as K

# Constants configurations
WIDTH, HEIGHT, SIZE = 640, 480, 28
CONFIDENCE_THRESHOLD = 0.9
FRAME_SKIP = 5 
PADDING = 15

# Initialize webcam
print("Starting webcam...")
cp = cv2.VideoCapture(0)
cp.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cp.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# Determine the input shape for the model based on the Keras backend
if K.image_data_format() == 'channels_first':
    input_shape = (1, SIZE, SIZE)
    first_dim, second_dim = 0, 1
else:
    input_shape = (SIZE, SIZE, 1)
    first_dim, second_dim = 0, 3

# Function to annotate the frame with the predicted label
def annotate(frame, label, location=(20, 30)):
    cv2.putText(frame, label, location, cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6, color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)

# Function to extract and preprocess a digit from the frame based on a bounding rectangle
def extract_digit(frame, rect, pad=PADDING):
    x, y, w, h = rect
    cropped_digit = final_img[y-pad:y+h+pad, x-pad:x+w+pad] / 255.0

    if cropped_digit.shape[0] >= 32 and cropped_digit.shape[1] >= 32:
        return cv2.resize(cropped_digit, (SIZE, SIZE))
    return None

# Function to preprocess the frame for MNIST-style classification
def img_to_mnist(frame):
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    return cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, blockSize=321, C=SIZE)

# Load the pre-trained MNIST model
print("Loading model...")
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)
model = load_model("mnist_cnn_model.h5")

# Define a dictionary mapping class indices to digit labels
labelz = dict(enumerate(["zero", "one", "two", "three", "four",
                         "five", "six", "seven", "eight", "nine"]))

frame_count = 0
while True:
    ret, frame = cp.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:  # Skip frames to reduce processing load
        continue

    # Preprocess the current frame
    final_img = img_to_mnist(frame)
    image_shown = frame.copy()  # Create a copy for annotations

    # Find contours in the processed frame
    contours, _ = cv2.findContours(final_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50]

    # Process each detected rectangle (bounding box)
    for rect in rects:
        x, y, w, h = rect
        mnist_frame = extract_digit(frame, rect)

        if mnist_frame is not None:
            # Prepare the digit for the model
            mnist_frame = np.expand_dims(mnist_frame, first_dim)
            mnist_frame = np.expand_dims(mnist_frame, second_dim)

            # Make predictions using the pre-trained model
            predictions = model.predict(mnist_frame, verbose=False)
            class_prediction = np.argmax(predictions, axis=-1)[0]  # Get the predicted class
            confidence = np.max(predictions)  # Get the prediction confidence

            if confidence >= CONFIDENCE_THRESHOLD:
                label = f"{labelz[class_prediction]} ({confidence:.2f})"
                annotate(image_shown, label, location=(x, y - 10))
                # Draw a bounding box around the detected digit
                cv2.rectangle(image_shown, (x - PADDING, y - PADDING),
                              (x + w + PADDING, y + h + PADDING), color=(0, 255, 0), thickness=2)

    # Show the annotated frame in a window
    cv2.imshow('Webcam', image_shown)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cp.release()
cv2.destroyAllWindows()
