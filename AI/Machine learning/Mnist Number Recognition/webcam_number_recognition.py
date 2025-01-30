import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

# Load the model
print("Loading the model...")
model = load_model("mnist_cnn_model.h5")

# Start the webcam
print("Starting the webcam...")
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a rectangle around the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        digit = gray[y:y+h, x:x+w]
        
        # Resize and normalize the number
        resized = cv2.resize(digit, (28, 28))
        normalized = resized / 255.0
        reshaped = normalized.reshape(1, 28, 28, 1)  # Add a batch dimension
        
        # Prediction
        prediction = model.predict(reshaped)
        predicted_class = np.argmax(prediction)
        
        # Draw the rectangle around the number in the original frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Predicted: {predicted_class}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with the prediction
    cv2.imshow("Webcam", frame)

    # Exit by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
