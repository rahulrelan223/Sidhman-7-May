from flask import Flask, request, jsonify, render_template
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from scipy.spatial.distance import cosine
import cv2

app = Flask(__name__)

def create_feature_extractor(input_shape=(224, 224, 3)):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    global_avg_pooling_layer = GlobalAveragePooling2D()(base_model.output)
    feature_extractor = Model(inputs=base_model.input, outputs=global_avg_pooling_layer)
    return feature_extractor

# Load the feature extractor model
input_shape = (224, 224, 3)
feature_extractor = create_feature_extractor(input_shape)

# Load the features and image names from the dataset
dataset_path = r'C:\Users\rahul\Desktop\SIdhman-google\new\dataset\Test'
loaded_data = np.load(r'extracted_features.npz')
dataset_features = loaded_data['features']
dataset_image_names = loaded_data['image_names']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/similar_images', methods=['POST'])
def find_similar_images():
    try:
        # Check if the post request has the file part
        if 'files[]' not in request.files:
            return jsonify({'error': 'No file part'})

        # Get uploaded image files
        uploaded_files = request.files.getlist('files[]')

        # If no files are selected
        if len(uploaded_files) == 0:
            return jsonify({'error': 'No files selected'})

        # Process uploaded images in batch
        batch_features = []
        for uploaded_file in uploaded_files:
            # Save the uploaded file to a temporary location
            uploaded_file_path = 'uploaded_image.jpg'
            uploaded_file.save(uploaded_file_path)

            # Load and preprocess the uploaded image
            img = cv2.imread(uploaded_file_path)
            img = cv2.resize(img, (224, 224))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            # Extract features from the uploaded image
            uploaded_image_features = feature_extractor.predict(img)

            # Ensure uploaded_image_features is a 1-D array
            uploaded_image_features = np.squeeze(uploaded_image_features)
            batch_features.append(uploaded_image_features)

            # Delete temporary file
            os.remove(uploaded_file_path)

        # Convert batch features to numpy array
        batch_features = np.array(batch_features)

        # Ensure dataset_features is a 2-D array
        dataset_features_squeezed = np.squeeze(dataset_features)

        # Compute cosine similarity with batch features
        similarities = 1 - cosine(batch_features, dataset_features_squeezed, axis=1)
        max_similarity_indices = np.argmax(similarities, axis=1)
        similar_image_names = [dataset_image_names[index] for index in max_similarity_indices]

        # Pass the paths of the similar images
        similar_image_paths = [os.path.join(dataset_path, name) for name in similar_image_names]

        # Prepare response data
        response_data = {
            'similar_image_names': similar_image_names,
            'similar_image_paths': similar_image_paths
        }

        # Render result.html template with response data
        return render_template('result.html', data=response_data)

    except Exception as e:
        # Error handling: Log any errors that occur
        print("An error occurred:", e)
        return jsonify({'error': 'An error occurred. Please try again later.'})

if __name__ == '__main__':
    app.run(debug=True)
