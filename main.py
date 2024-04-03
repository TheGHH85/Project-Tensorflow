import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models

class_name = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Load the trained model
model = models.load_model('model.h5')

# Load and preprocess the image
img = cv.imread('plane.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (32, 32))  # Resize the image to match the model's input shape
img = img / 255.0  # Normalize the pixel values

# Display the image
plt.imshow(img, cmap=plt.cm.binary)
plt.show()

# Make prediction
prediction = model.predict(np.expand_dims(img, axis=0))
index = np.argmax(prediction)
print(f"Prediction: {class_name[index]}")
