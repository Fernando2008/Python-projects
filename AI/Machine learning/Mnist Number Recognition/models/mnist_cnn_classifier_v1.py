import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tensorflow.keras.datasets import mnist # type: ignore

# 1. Loading and pre-processing of the MNIST dataset
print("Loading the MNIST dataset...")
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalization (from 0-255 to 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Add a shape per channel (richiesto da CNN)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Converts labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# 2. CNN creation
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # First convolutional layer
    MaxPooling2D((2, 2)),  # First pooling layer
    Conv2D(64, (3, 3), activation='relu'),  # Second convolutional layer
    MaxPooling2D((2, 2)),  # Second pooling layer
    Flatten(),  # Flatten the data
    Dense(128, activation='relu'),  # Dense layer fully connected
    Dense(10, activation='softmax')  # Output layer with 10 classes
])

# 3. Template
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Training the model
print("Training the model...")
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# 5. Model valutation
print("Model valutation...")
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test set accuracy: {test_accuracy * 100:.2f}%")

# Test set predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 6. Display some predictions
def plot_predictions(images, true_labels, pred_labels, num_images=10):
    random_indices = random.sample(range(len(images)), num_images)
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(random_indices):
        plt.subplot(1, num_images, i+1)
        plt.imshow(images[idx].reshape(28, 28), cmap="gray")
        plt.title(f"Pred: {pred_labels[idx]}\nTrue: {true_labels[idx]}")
        plt.axis("off")
    plt.show()

plot_predictions(X_test, y_test_classes, y_pred_classes)

# Saving the model as .h5 file
model.save("mnist_cnn_model.h5")
print("Model saved as mnist_cnn_model.h5")