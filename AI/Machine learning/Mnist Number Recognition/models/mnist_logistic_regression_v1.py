import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Load the MNIST dataset
print("Loading the MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# 2. Convert the labels (y) to integers (since they are stored as strings)
y = y.astype(np.uint8)

# 3. Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% for training, 20% for testing

# 4. Normalize the pixel values (scale them to range 0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# 5. Convert DataFrames to NumPy arrays for compatibility with sklearn models
X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

# 6. Train the Logistic Regression model
print("Training the model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Make predictions on the test set
y_pred = model.predict(X_test)

# 8. Calculate and print the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")

# 9. Create and display the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 10. Function to display a random selection of predicted vs true labels
def plot_predictions(images, labels, predictions, num_images=10):
    random_indices = random.sample(range(len(images)), num_images)
    plt.figure(figsize=(10, 5))
    for i, idx in enumerate(random_indices):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[idx].reshape(28, 28), cmap="gray")
        plt.title(f"Pred: {predictions[idx]}\nTrue: {labels[idx]}")
        plt.axis("off")
    plt.show()

plot_predictions(X_test, y_test, y_pred)
