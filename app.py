import numpy as np
import os
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from scipy.spatial.distance import cosine

app = Flask(__name__)

# Function to create the feature extractor model
def create_feature_extractor(input_shape=(224, 224, 3)):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    global_avg_pooling_layer = GlobalAveragePooling2D()(base_model.output)
    feature_extractor = Model(inputs=base_model.input, outputs=global_avg_pooling_layer)
    return feature_extractor

# Load the feature extractor model
feature_extractor = create_feature_extractor()

# Define the paths for the dataset
dataset_path = 'Test'  # Assuming the dataset folder is in the root directory of your GitHub repo
class_names = ['gear', 'pin']  # Update with your actual class names

# Load the features and image names from the dataset
features = []
image_names = []

for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    image_files = os.listdir(class_path)
    for image_file in image_files:
        image_path = os.path.join(class_path, image_file)
        # Load and preprocess the image
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        # Extract features from the image
        image_features = feature_extractor.predict(img_array)
        # Ensure image_features is a 1-D array
        image_features = np.squeeze(image_features)
        # Append the features and image name to the lists
        features.append(image_features)
        image_names.append(image_file)

# Convert features and image names to numpy arrays
dataset_features = np.array(features)
dataset_image_names = np.array(image_names)

@app.route('/', methods=['POST'])
def upload_image():
    try:
        # Check if the request contains an image file
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        # Get uploaded image file
        uploaded_file = request.files['file']

        # If no file is selected, return an error
        if uploaded_file.filename == '':
            return jsonify({'error': 'No selected file'})

        # Save the uploaded file to a temporary location
        uploaded_file_path = 'uploaded_image.jpg'
        uploaded_file.save(uploaded_file_path)

        # Load and preprocess the uploaded image
        img = tf.keras.preprocessing.image.load_img(uploaded_file_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # Extract features from the uploaded image
        uploaded_image_features = feature_extractor.predict(img_array)

        # Ensure uploaded_image_features is a 1-D array
        uploaded_image_features = np.squeeze(uploaded_image_features)

        # Ensure dataset_features is a 2-D array
        dataset_features_squeezed = np.squeeze(dataset_features)

        # Find similar images by computing cosine similarity
        similarities = [1 - cosine(uploaded_image_features, feat) for feat in dataset_features_squeezed]
        max_similarity_index = np.argmax(similarities)
        similar_image_name = dataset_image_names[max_similarity_index]

        # Pass the path of the similar image as a response
        similar_image_path = os.path.join(dataset_path, similar_image_name)

        # Return the similar image path in the response
        return jsonify({'similar_image_path': similar_image_path})

    except Exception as e:
        # Error handling: Log any errors that occur
        print("An error occurred:", e)
        return jsonify({'error': 'An error occurred. Please try again later.'})

if __name__ == '__main__':
    app.run(debug=True)
