import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import cv2
from PIL import Image
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Set seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define paths and parameters
BASE_DIR = 'data/Data'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
IMG_SIZE = 224  # Keep this the same for compatibility
BATCH_SIZE = 64
NUM_WORKERS = 4

# Function to extract handcrafted features from images
def extract_features(img_path):
    # Read image and convert to grayscale
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error reading image: {img_path}")
        return None
        
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []
    
    # 1. Basic statistical features
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    features.extend(hist)  # Add histogram features
    
    # Statistical moments
    mean = np.mean(gray)
    std = np.std(gray)
    skewness = np.mean(((gray - mean)/std)**3) if std > 0 else 0
    kurtosis = np.mean(((gray - mean)/std)**4) if std > 0 else 0
    
    features.extend([mean, std, skewness, kurtosis])
    
    # 2. Haralick texture features (GLCM)
    glcm = graycomatrix(gray, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        symmetric=True, normed=True)
    
    glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in glcm_props:
        features.extend(graycoprops(glcm, prop).flatten())
    
    # 3. Local Binary Patterns for texture
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=n_points+2, range=(0, n_points+2), density=True)
    features.extend(lbp_hist)
    
    # Add another LBP with different radius for multi-scale analysis
    radius2 = 2
    n_points2 = 8 * radius2
    lbp2 = local_binary_pattern(gray, n_points2, radius2, method='uniform')
    lbp_hist2, _ = np.histogram(lbp2, bins=n_points2+2, range=(0, n_points2+2), density=True)
    features.extend(lbp_hist2)
    
    # 4. Histogram of Oriented Gradients (HOG) features
    hog_features, _ = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), visualize=True, feature_vector=True)
    
    # Take a subset of HOG features to reduce dimensionality
    hog_features_subset = hog_features[::5]  # Take every 5th feature instead of 10th
    features.extend(hog_features_subset)
    
    # 5. Shape and edge features
    # Canny edge detection with multiple thresholds
    edges1 = cv2.Canny(gray, 50, 150)
    edges2 = cv2.Canny(gray, 100, 200)
    edge_density1 = np.sum(edges1 > 0) / (IMG_SIZE * IMG_SIZE)
    edge_density2 = np.sum(edges2 > 0) / (IMG_SIZE * IMG_SIZE)
    features.extend([edge_density1, edge_density2])
    
    # Add Fourier transform features for frequency analysis
    f_transform = np.fft.fft2(gray)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
    
    # Extract statistical features from the magnitude spectrum
    mag_mean = np.mean(magnitude_spectrum)
    mag_std = np.std(magnitude_spectrum)
    mag_median = np.median(magnitude_spectrum)
    mag_max = np.max(magnitude_spectrum)
    features.extend([mag_mean, mag_std, mag_median, mag_max])
    
    # Add color features from original image
    for i in range(3):  # For each color channel
        channel = img[:,:,i]
        features.extend([np.mean(channel), np.std(channel), np.median(channel)])
    
    return np.array(features)

# Function to extract features from a directory of images
def extract_features_from_directory(directory, class_names):
    features = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found. Skipping.")
            continue
            
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing {len(image_files)} images from {class_name}")
        
        for img_file in tqdm(image_files):
            img_path = os.path.join(class_dir, img_file)
            img_features = extract_features(img_path)
            
            if img_features is not None:
                features.append(img_features)
                labels.append(class_idx)
    
    return np.array(features), np.array(labels)

# Train CV-ML model
def train_cv_model():
    # Get class names from the directory structure
    train_class_names = sorted(os.listdir(TRAIN_DIR))
    test_class_names = sorted(os.listdir(TEST_DIR))
    print(f"Train Classes: {train_class_names}")
    print(f"Test Classes: {test_class_names}")
    
    # Check for differences in class names
    if set(train_class_names) != set(test_class_names):
        print("WARNING: Train and test sets have different class names!")
        print(f"Train set has: {set(train_class_names) - set(test_class_names)}")
        print(f"Test set has: {set(test_class_names) - set(train_class_names)}")
    
    # Use only classes that appear in both train and test sets
    class_names = sorted(list(set(train_class_names).intersection(set(test_class_names))))
    print(f"Using common classes: {class_names}")
    
    # Extract features from training and test sets
    print("Extracting features from training set...")
    train_features, train_labels = extract_features_from_directory(TRAIN_DIR, class_names)
    
    print("Extracting features from test set...")
    test_features, test_labels = extract_features_from_directory(TEST_DIR, class_names)
    
    print(f"Training features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")
    
    # Split training data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    test_features_scaled = scaler.transform(test_features)
    
    # Train multiple models for ensemble learning
    
    # 1. Random Forest Classifier with improved parameters
    print("Training Random Forest classifier...")
    rf_clf = RandomForestClassifier(
        n_estimators=300,  # Increased from 200
        max_depth=25,      # Increased from 20
        min_samples_split=4,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train_scaled, y_train)
    
    # 2. Support Vector Machine with improved parameters
    print("Training SVM classifier...")
    svm_clf = SVC(
        C=20,  # Increased from 10
        kernel='rbf', 
        gamma='auto',  # Changed from 'scale'
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    svm_clf.fit(X_train_scaled, y_train)
    
    # 3. K-Nearest Neighbors with improved parameters
    print("Training KNN classifier...")
    knn_clf = KNeighborsClassifier(
        n_neighbors=5,  # Changed from 7
        weights='distance',
        algorithm='auto',
        p=2,
        n_jobs=-1
    )
    knn_clf.fit(X_train_scaled, y_train)
    
    # 4. Add Gradient Boosting Classifier
    from sklearn.ensemble import GradientBoostingClassifier
    print("Training Gradient Boosting classifier...")
    gb_clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    gb_clf.fit(X_train_scaled, y_train)
    
    # Create a voting classifier for ensemble prediction with weights
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf_clf),
            ('svm', svm_clf),
            ('knn', knn_clf),
            ('gb', gb_clf)
        ],
        voting='soft',  # Use probability estimates for prediction
        weights=[2, 1, 1, 2]  # Give more weight to RF and GB
    )
    
    print("Training ensemble model...")
    voting_clf.fit(X_train_scaled, y_train)
    
    # Evaluate on validation set
    val_preds = voting_clf.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Validation accuracy: {val_acc:.4f}")
    
    # Evaluate on test set
    test_preds = voting_clf.predict(test_features_scaled)
    test_acc = accuracy_score(test_labels, test_preds)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save the model and preprocessing components
    model_data = {
        'ensemble': voting_clf,
        'rf': rf_clf,
        'svm': svm_clf,
        'knn': knn_clf,
        'gb': gb_clf,
        'scaler': scaler,
        'class_names': class_names
    }
    
    with open('models/cv_ensemble_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("CV model trained and saved successfully!")
    
    # Generate classification report
    print("\nClassification Report (Test Set):")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Print confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Calculate per-class accuracy
    print('\nPer-class Accuracy:')
    for i, class_name in enumerate(class_names):
        class_indices = np.where(test_labels == i)[0]
        if len(class_indices) > 0:
            class_correct = np.sum(test_preds[class_indices] == i)
            class_acc = class_correct / len(class_indices)
            print(f'{class_name}: {class_acc:.4f} ({class_correct}/{len(class_indices)})')
    
    return test_acc

# Entry point
if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Train the CV-ML model
    accuracy = train_cv_model()
    print(f"Final accuracy: {accuracy:.4f}")