import os
import numpy as np
import cv2
import pickle
import argparse
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm

# Function to extract handcrafted features from images (must match the training feature extraction)
def extract_features(img_path, img_size=224):
    # Read image and convert to grayscale
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading image: {img_path}")
        return None
        
    img = cv2.resize(img, (img_size, img_size))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []
    
    # 1. Basic statistical features
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    
    # Statistical moments
    mean = np.mean(gray)
    std = np.std(gray)
    skewness = np.mean(((gray - mean)/std)**3) if std > 0 else 0
    kurtosis = np.mean(((gray - mean)/std)**4) if std > 0 else 0
    
    features.extend([mean, std, skewness, kurtosis])
    
    # 2. Haralick texture features (GLCM)
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        symmetric=True, normed=True)
    
    glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    for prop in glcm_props:
        features.extend(graycoprops(glcm, prop).flatten())
    
    # 3. Local Binary Patterns for texture
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=n_points+2, range=(0, n_points+2), density=True)
    features.extend(lbp_hist)
    
    # 4. Histogram of Oriented Gradients (HOG) features
    # Use a smaller cell size for computational efficiency
    hog_features, _ = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), visualize=True, feature_vector=True)
    
    # Take a subset of HOG features to reduce dimensionality
    hog_features_subset = hog_features[::10]  # Take every 10th feature
    features.extend(hog_features_subset)
    
    # 5. Shape and edge features
    # Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (img_size * img_size)
    features.append(edge_density)
    
    # Add Fourier transform features for frequency analysis
    f_transform = np.fft.fft2(gray)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
    
    # Extract statistical features from the magnitude spectrum
    mag_mean = np.mean(magnitude_spectrum)
    mag_std = np.std(magnitude_spectrum)
    features.extend([mag_mean, mag_std])
    
    return np.array(features)

# Function to load the model and its components
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    ensemble = model_data['ensemble']
    scaler = model_data['scaler']
    class_names = model_data['class_names']
    
    return ensemble, scaler, class_names

# Function to predict a single image
def predict_image(image_path, model, scaler, class_names):
    # Extract features from the image
    features = extract_features(image_path)
    
    if features is None:
        return None, 0
    
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Get the predicted class and probability
    predicted_class = class_names[prediction]
    confidence = probabilities[prediction]
    
    # Create a dictionary of all class probabilities
    all_probs = {class_names[i]: prob for i, prob in enumerate(probabilities)}
    
    return predicted_class, confidence, all_probs

# Function to predict images in a folder
def predict_folder(folder_path, model, scaler, class_names):
    # Get all image files in the folder
    image_files = [f for f in os.listdir(folder_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    results = []
    for img_file in tqdm(image_files):
        img_path = os.path.join(folder_path, img_file)
        predicted_class, confidence, _ = predict_image(img_path, model, scaler, class_names)
        
        if predicted_class is not None:
            results.append((img_file, predicted_class, confidence))
    
    return results

# Function to evaluate model on test set
def evaluate_test_set(test_dir, model, scaler, class_names):
    all_preds = []
    all_labels = []
    
    # Dictionary to store class-wise accuracy
    class_predictions = {class_name: {'correct': 0, 'total': 0} for class_name in class_names}
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found. Skipping.")
            continue
            
        # Get all image files in the class directory
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Evaluating {len(image_files)} images from class: {class_name}")
        
        for img_file in tqdm(image_files):
            img_path = os.path.join(class_dir, img_file)
            predicted_class, _, _ = predict_image(img_path, model, scaler, class_names)
            
            if predicted_class is not None:
                # Update statistics
                class_predictions[class_name]['total'] += 1
                if predicted_class == class_name:
                    class_predictions[class_name]['correct'] += 1
                
                # Track predictions for metrics
                all_preds.append(class_names.index(predicted_class))
                all_labels.append(class_idx)
    
    # Calculate overall accuracy
    overall_accuracy = accuracy_score(all_labels, all_preds)
    
    # Calculate class-wise accuracy
    class_accuracies = {}
    for class_name, stats in class_predictions.items():
        if stats['total'] > 0:
            class_accuracies[class_name] = stats['correct'] / stats['total']
        else:
            class_accuracies[class_name] = 0
    
    return overall_accuracy, class_accuracies, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser(description='CV-ML Model Inference for Lung Cancer Classification')
    parser.add_argument('--model', type=str, default='models/cv_ensemble_model.pkl', 
                        help='Path to the model file')
    parser.add_argument('--image', type=str, help='Path to a single image for prediction')
    parser.add_argument('--folder', type=str, help='Path to a folder of images for prediction')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate on the test set')
    parser.add_argument('--test_dir', type=str, default='data/Data/test', 
                        help='Path to the test directory')
    
    args = parser.parse_args()
    
    # Load the model
    model, scaler, class_names = load_model(args.model)
    print(f"Model loaded. Number of classes: {len(class_names)}")
    
    # Single image prediction
    if args.image:
        predicted_class, confidence, all_probs = predict_image(args.image, model, scaler, class_names)
        print(f"Image: {args.image}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}")
        
        # Print all class probabilities
        print("\nClass probabilities:")
        for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"{cls}: {prob:.4f}")
    
    # Folder prediction
    elif args.folder:
        results = predict_folder(args.folder, model, scaler, class_names)
        print(f"Predicted {len(results)} images in folder: {args.folder}")
        for img_file, predicted_class, confidence in results:
            print(f"{img_file}: {predicted_class} (confidence: {confidence:.2f})")
    
    # Evaluate on test set
    elif args.evaluate:
        print(f"Evaluating on test set: {args.test_dir}")
        overall_accuracy, class_accuracies, all_preds, all_labels = evaluate_test_set(
            args.test_dir, model, scaler, class_names
        )
        
        print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
        print("\nClass Accuracies:")
        for class_name, accuracy in class_accuracies.items():
            print(f"{class_name}: {accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        # Print confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion Matrix:")
        print(cm)
    
    # Default: evaluate on test set
    else:
        print(f"Evaluating on test set: {args.test_dir}")
        overall_accuracy, class_accuracies, all_preds, all_labels = evaluate_test_set(
            args.test_dir, model, scaler, class_names
        )
        
        print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
        print("\nClass Accuracies:")
        for class_name, accuracy in class_accuracies.items():
            print(f"{class_name}: {accuracy:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == "__main__":
    main()