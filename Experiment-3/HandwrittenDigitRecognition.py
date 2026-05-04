import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

# Load MNIST dataset
mnist = tf.keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build Neural Network
model = Sequential([
    Flatten(input_shape=(28,28)),   # Convert 2D image to 1D
    Dense(128, activation='relu'),  # Hidden layer
    Dense(10, activation='softmax') # Output layer (digits 0-9)
])

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train, y_train, epochs=5)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", accuracy)

# Predict a digit
prediction = model.predict(X_test)

print("\nPredicted Digit:", prediction[0].argmax())
print("Actual Digit:", y_test[0])

# Display the image
plt.imshow(X_test[0], cmap="gray")
plt.title("Handwritten Digit")
plt.show()

plt.imshow(X_test[0], cmap='gray')
plt.title(f"Actual Digit: {y_test[0]}")
plt.show()