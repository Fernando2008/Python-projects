import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

# Load the model
print("Loading the model...")
model = load_model("mnist_cnn_model.h5")

# Start the webcam
print("Starting the webcam...")
cap = cv2.VideoCapture(0)

# Webcam configuration
WIDTH, HEIGHT, FPS = 1280, 720, 30
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

# Define the prediction function
def prediction(image, model):
    # Preprocess the image
    img = cv2.resize(image, (28, 28))
    img = cv2.bitwise_not(img)
    img = img / 255
    img = img.reshape(1, 28, 28, 1)

    # Make the prediction
    predict = model.predict(img)
    prob = np.amax(predict)
    result = np.argmax(predict)

    # Check confidence threshold
    if prob < 0.75:
        result = -1
        prob = 0

    return result, prob

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Define the bounding box rectangle
    bbox_size = (60, 60)
    bbox = [
        (int(HEIGHT // 2 - bbox_size[0] // 2), int(HEIGHT // 2 - bbox_size[1] // 2)),
        (int(WIDTH // 2 + bbox_size[0] // 2), int(WIDTH // 2 + bbox_size[1] // 2))
    ]

    # Crop and preprocess the image from the webcam
    img_cropped = frame[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.bitwise_not(img_gray)
    img_gray = cv2.resize(img_gray, (200, 200))
    cv2.imshow("Cropped", img_gray)

    # Make the prediction
    result, probability = prediction(img_gray, model)

    # Display the prediction and probability
    cv2.putText(frame, f"Prediction: {result}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Probability: " + "{:.2f}".format(probability), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Change the bounding box color based on confidence
    color = (0, 255, 0) if probability >= 0.75 else (0, 0, 255)
    cv2.rectangle(frame, bbox[0], bbox[1], color, 3)  # Draw the bounding box

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
