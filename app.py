from flask import Flask, request, jsonify
import pickle
import numpy as np
import cv2 as cv
import os
from skimage.feature import local_binary_pattern
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Load KNN model
model_path = 'knn_model3.pkl'
try:
    with open(model_path, 'rb') as model_file:
        knn_model = joblib.load(model_file)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    knn_model = None

# Directory to save uploaded images
UPLOAD_FOLDER = 'static/images/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to resize the image before processing
def resize_image(image, target_size=(128, 128)):
    return cv.resize(image, target_size)

# Function to crop the image and extract the mouth region
def crop_image(image_path):
    output_folder = 'static/cropped/'
    os.makedirs(output_folder, exist_ok=True)

    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv.imread(image_path)

    if image is None:
        print(f"Image at {image_path} not found.")
        return None

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

    if len(faces) == 0:
        print("No faces detected in the image.")
        return None

    for (x, y, w, h) in faces:
        lip_top = int(y + h * 0.65)
        lip_bottom = int(y + h * 0.9)
        lip_left = x
        lip_right = x + w

        lip_top = max(0, lip_top)
        lip_bottom = min(image.shape[0], lip_bottom)
        lip_left = max(0, lip_left)
        lip_right = min(image.shape[1], lip_right)

        mouth_roi = image[lip_top:lip_bottom, lip_left:lip_right]
        output_path = os.path.join(output_folder, 'cropped_image.jpeg')
        cv.imwrite(output_path, mouth_roi)
        return output_path
    return None

# Function to extract LBP features from an image
def extract_lbp_features(image_path, radius=2, n_points=8*2, method='uniform'):
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if image is None:
        return None
    lbp_image = local_binary_pattern(image, n_points, radius, method)
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

# Function to extract Gabor features
def extract_gabor_features(image_path):
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Image at {image_path} not found.")
        return None

    # Define Gabor parameters
    kernels = []
    for theta in range(4):  # 4 orientations
        theta = theta / 4. * np.pi  # Convert to radians
        for sigma in (1, 3):  # 2 scales
            for lamda in np.pi / 2 * np.arange(0, 2):  # 2 wavelengths
                gabor = cv.getGaborKernel((21, 21), sigma, theta, lamda, 0.5, 0, ktype=cv.CV_32F)
                kernels.append(gabor)

    # Apply Gabor filters to the image
    features = []
    for kernel in kernels:
        filtered = cv.filter2D(image, cv.CV_8UC3, kernel)
        features.append(np.mean(filtered))  # You can extract more features if needed

    return np.array(features)

# Function to extract RGB histogram features
def extract_rgb_histogram(image_path):
    image = cv.imread(image_path)
    if image is None:
        print(f"Image at {image_path} not found.")
        return None

    # Extract histogram for each channel and normalize
    chans = cv.split(image)
    features = []
    for chan in chans:
        hist = cv.calcHist([chan], [0], None, [256], [0, 256])
        hist = cv.normalize(hist, hist).flatten()
        features.extend(hist)

    return np.array(features)

# Menggabungkan semua fitur (LBP, Gabor, RGB histogram)
def extract_features(image_path):
    lbp_features = extract_lbp_features(image_path)
    gabor_features = extract_gabor_features(image_path)
    rgb_histogram_features = extract_rgb_histogram(image_path)

    # Menggabungkan semua fitur menjadi satu vektor fitur
    if lbp_features is not None and gabor_features is not None and rgb_histogram_features is not None:
        return np.concatenate((lbp_features, gabor_features, rgb_histogram_features))
    else:
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_file = request.files.get('image')
        if image_file:
            image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
            image_file.save(image_path)

            cropped_image_path = crop_image(image_path)
            if cropped_image_path:
                # Menggunakan fitur yang diekstrak untuk prediksi
                features = extract_features(cropped_image_path)
                if features is not None and features.shape[0] == 802:
                    prediction = knn_model.predict([features])
                    result = 'smoker' if  int(prediction[0]) == 2 else 'non-smoker' if  int(prediction[0]) == 1 else 'unknown'
                    return jsonify({'prediction': result})
                    # return jsonify({'prediction': int(prediction[0])})
                else:
                    return jsonify({'error': 'Invalid feature length or extraction failed.'})
                
            else:
                return jsonify({'error': 'No face detected'})
        else:
            return jsonify({'error': 'No image uploaded'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
