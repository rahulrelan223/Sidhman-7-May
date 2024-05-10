from flask import Flask, request, render_template
import numpy as np
import os
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

# Load the features and image names from the dataset
dataset_path = r'Test'
loaded_data = np.load(r'extracted_features.npz')
dataset_features = loaded_data['features']
dataset_image_names = loaded_data['image_names']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Check if the post request has the file part
            if 'file' not in request.files:
                return "No file part"
        
            # Get uploaded image file
            uploaded_file = request.files['file']
        
            # If the user does not select a file, the browser submits an empty file without a filename
            if uploaded_file.filename == '':
                return "No selected file"
        
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
            
            # Pass the path of the similar image to the template
            similar_image_path = os.path.join(dataset_path, similar_image_name)
            
            # Create similar images data
            similar_images = {
                'name': similar_image_name,
                'path': similar_image_path
            }
            
            # Print the path of the similar image along with its name
            print("Similar Image Name:", similar_image_name)
            print("Similar Image Path:", similar_image_path)
            
            return render_template('result.html', similar_images=similar_images)
            
        except Exception as e:
            # Error handling: Log any errors that occur
            print("An error occurred:", e)
            return "An error occurred. Please try again later."

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
