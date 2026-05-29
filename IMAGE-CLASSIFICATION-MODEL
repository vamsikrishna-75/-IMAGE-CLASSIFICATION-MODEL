# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

print("TensorFlow Version:", tf.__version__)

# =========================================
# 2. LOAD DATASET (CIFAR-10)
# =========================================
# 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values (0–255 → 0–1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Class names
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

print("Training data shape:", X_train.shape)

# =========================================
# 3. VISUALIZE SAMPLE IMAGES
# =========================================
plt.figure(figsize=(8,8))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')
plt.show()

# =========================================
# 4. BUILD CNN MODEL
# =========================================
model = models.Sequential([
    
    # Convolution Layer 1
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),
    
    # Convolution Layer 2
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    # Convolution Layer 3
    layers.Conv2D(64, (3,3), activation='relu'),
    
    # Flatten
    layers.Flatten(),
    
    # Fully Connected Layers
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')   # 10 classes
])

# =========================================
# 5. COMPILE MODEL
# =========================================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================================
# 6. TRAIN MODEL
# =========================================
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

# =========================================
# 7. EVALUATE MODEL
# =========================================
test_loss, test_acc = model.evaluate(X_test, y_test)

print("\nTest Accuracy:", test_acc)

# =========================================
# 8. PLOT ACCURACY GRAPH
# =========================================
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.show()

# =========================================
# 9. PREDICTIONS
# =========================================
predictions = model.predict(X_test)

def predict_image(index):
    plt.imshow(X_test[index])
    plt.title("Predicted: " + class_names[np.argmax(predictions[index])])
    plt.axis('off')
    plt.show()

# Test prediction
predict_image(5)

# =========================================
# 10. SAVE MODEL
# =========================================
model.save("cnn_image_classifier.h5")

print("Model saved successfully!")
