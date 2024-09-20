import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Update with the correct path to your model file
model_path = 'C:/Users/Lenovo/image_detector/your_model.keras'
model = load_model(model_path)

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} not found.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (150, 150))  # Match the size used during training
    image = np.expand_dims(image, axis=0) / 255.0
    return image

# Function to make predictions
def predict(image_path):
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    class_index = np.argmax(prediction, axis=-1)  # Get the index of the max probability
    return class_index[0]

# Function to display image and prediction
def display_image_and_prediction(image_path, prediction):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} not found.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(f"Predicted class: {prediction}")
    plt.axis('off')
    plt.show()

# Main function
if __name__ == "__main__":
    image_path = 'C:/Users/Lenovo/image_detector/dataset/test/test_image.jpg'  # Update with the correct path to your test image
    result = predict(image_path)
    print(f"Predicted class: {result}")
    display_image_and_prediction(image_path, result)
