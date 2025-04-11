import streamlit as st
import os
import numpy as np
import cv2
import pickle
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import tempfile
from tqdm import tqdm

# Set page configuration with dark theme
st.set_page_config(
    page_title="Lung Cancer Classification",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Apply dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Define paths
DATA_DIR = "/home/ubuntu/eureka/bugman/sandboxes/cd9e41b9-64d3-499e-9fa0-90d732cf6bf4/data"
MODELS_DIR = "/home/ubuntu/eureka/bugman/sandboxes/cd9e41b9-64d3-499e-9fa0-90d732cf6bf4/snapshots/efa2404339822b29419feaf97e8dea033a156eb3/models"
MODEL_PATH = os.path.join(MODELS_DIR, "cv_ensemble_model.pkl")
TEST_DIR = os.path.join(DATA_DIR, "Data/test")

# Function to extract handcrafted features from images
def extract_features(img, img_size=224):
    """
    Extract features from an image (either from path or image object)
    """
    # Handle different input types
    if isinstance(img, str):
        # Read image from path
        img = cv2.imread(img)
        if img is None:
            st.error(f"Error reading image")
            return None
    elif isinstance(img, np.ndarray):
        # Already an image array
        pass
    else:
        st.error("Unsupported image type")
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
@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        ensemble = model_data['ensemble']
        scaler = model_data['scaler']
        class_names = model_data['class_names']
        
        return ensemble, scaler, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Function to predict a single image
def predict_image(image, model, scaler, class_names):
    # Extract features from the image
    features = extract_features(image)
    
    if features is None:
        return None, 0, {}
    
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

# Function to evaluate model on test set
def evaluate_test_set(test_dir, model, scaler, class_names, progress_bar=None):
    all_preds = []
    all_labels = []
    results = []
    
    # Dictionary to store class-wise accuracy
    class_predictions = {class_name: {'correct': 0, 'total': 0} for class_name in class_names}
    
    # Get total number of images for progress bar
    total_images = 0
    for class_name in class_names:
        class_dir = os.path.join(test_dir, class_name)
        if os.path.exists(class_dir):
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_images += len(image_files)
    
    # Initialize progress bar if not provided
    if progress_bar is None:
        progress_bar = st.progress(0)
    
    processed_images = 0
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            st.warning(f"Directory {class_dir} not found. Skipping.")
            continue
            
        # Get all image files in the class directory
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for img_file in image_files:
            img_path = os.path.join(class_dir, img_file)
            predicted_class, confidence, all_probs = predict_image(img_path, model, scaler, class_names)
            
            if predicted_class is not None:
                # Update statistics
                class_predictions[class_name]['total'] += 1
                is_correct = predicted_class == class_name
                if is_correct:
                    class_predictions[class_name]['correct'] += 1
                
                # Track predictions for metrics
                all_preds.append(class_names.index(predicted_class))
                all_labels.append(class_idx)
                
                # Store result
                results.append({
                    'Image': img_file,
                    'True Class': class_name,
                    'Predicted Class': predicted_class,
                    'Confidence': confidence,
                    'Correct': is_correct
                })
            
            # Update progress
            processed_images += 1
            progress_bar.progress(processed_images / total_images)
    
    # Calculate overall accuracy
    overall_accuracy = accuracy_score(all_labels, all_preds)
    
    # Calculate class-wise accuracy
    class_accuracies = {}
    for class_name, stats in class_predictions.items():
        if stats['total'] > 0:
            class_accuracies[class_name] = stats['correct'] / stats['total']
        else:
            class_accuracies[class_name] = 0
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return overall_accuracy, class_accuracies, cm, class_names, results

# Function to convert OpenCV image to PIL format for display
def cv2_to_pil(cv2_img):
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)

# Function to plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

# Function to plot class accuracies
def plot_class_accuracies(class_accuracies):
    plt.figure(figsize=(12, 6))
    classes = list(class_accuracies.keys())
    accuracies = list(class_accuracies.values())
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]
    classes = [classes[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    
    bars = plt.bar(classes, accuracies, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Class-wise Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf

# Main app
def main():
    st.title("🫁 Lung Cancer Classification App")
    
    # Load model
    with st.spinner("Loading model..."):
        model, scaler, class_names = load_model(MODEL_PATH)
    
    if model is None:
        st.error("Failed to load model. Please check the model path.")
        return
    
    st.success(f"Model loaded successfully! Detected {len(class_names)} classes: {', '.join(class_names)}")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Single Image Prediction", "Batch Prediction", "Test Set Evaluation"])
    
    # Tab 1: Single Image Prediction
    with tab1:
        st.header("Predict a Single Image")
        
        # Image upload
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            uploaded_file.seek(0)  # Reset file pointer
            
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(cv2_to_pil(image), caption="Uploaded Image", width=300)
            
            # Make prediction
            if st.button("Predict", key="predict_single"):
                with st.spinner("Analyzing image..."):
                    predicted_class, confidence, all_probs = predict_image(image, model, scaler, class_names)
                
                if predicted_class is not None:
                    # Display prediction
                    st.success(f"Predicted Class: **{predicted_class}**")
                    st.info(f"Confidence: {confidence:.2f}")
                    
                    # Display all probabilities
                    st.subheader("Class Probabilities")
                    
                    # Create bar chart for probabilities
                    probs_df = pd.DataFrame({
                        'Class': list(all_probs.keys()),
                        'Probability': list(all_probs.values())
                    })
                    probs_df = probs_df.sort_values('Probability', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.bar(probs_df['Class'], probs_df['Probability'], color='skyblue')
                    ax.set_xlabel('Class')
                    ax.set_ylabel('Probability')
                    ax.set_title('Class Probabilities')
                    plt.xticks(rotation=45, ha='right')
                    
                    # Add value labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f'{height:.3f}', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
    
    # Tab 2: Batch Prediction
    with tab2:
        st.header("Batch Prediction")
        
        # Multiple file upload
        uploaded_files = st.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} images")
            
            if st.button("Predict All", key="predict_batch"):
                results = []
                
                # Create progress bar
                progress_bar = st.progress(0)
                
                for i, uploaded_file in enumerate(uploaded_files):
                    # Read image
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    # Make prediction
                    predicted_class, confidence, _ = predict_image(image, model, scaler, class_names)
                    
                    if predicted_class is not None:
                        results.append({
                            'Image': uploaded_file.name,
                            'Predicted Class': predicted_class,
                            'Confidence': confidence
                        })
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Display results
                if results:
                    st.subheader("Prediction Results")
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
                    
                    # Download results as CSV
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="batch_prediction_results.csv",
                        mime="text/csv"
                    )
                    
                    # Display summary
                    st.subheader("Prediction Summary")
                    class_counts = results_df['Predicted Class'].value_counts()
                    
                    # Create pie chart
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')
                    plt.title('Distribution of Predicted Classes')
                    st.pyplot(fig)
    
    # Tab 3: Test Set Evaluation
    with tab3:
        st.header("Test Set Evaluation")
        
        st.write(f"Test directory: {TEST_DIR}")
        
        # Check if test directory exists
        if not os.path.exists(TEST_DIR):
            st.error(f"Test directory not found: {TEST_DIR}")
        else:
            # Count images in test set
            total_images = 0
            class_counts = {}
            
            for class_name in class_names:
                class_dir = os.path.join(TEST_DIR, class_name)
                if os.path.exists(class_dir):
                    image_files = [f for f in os.listdir(class_dir) 
                                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    class_counts[class_name] = len(image_files)
                    total_images += len(image_files)
                else:
                    class_counts[class_name] = 0
            
            st.write(f"Found {total_images} images in test set")
            
            # Display class distribution
            st.subheader("Class Distribution in Test Set")
            class_dist_df = pd.DataFrame({
                'Class': list(class_counts.keys()),
                'Count': list(class_counts.values())
            })
            class_dist_df = class_dist_df.sort_values('Count', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(class_dist_df['Class'], class_dist_df['Count'], color='skyblue')
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title('Class Distribution in Test Set')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height}', ha='center', va='bottom')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Evaluate button
            if st.button("Evaluate on Test Set", key="evaluate_test"):
                with st.spinner("Evaluating on test set..."):
                    progress_bar = st.progress(0)
                    overall_accuracy, class_accuracies, cm, class_names, results = evaluate_test_set(
                        TEST_DIR, model, scaler, class_names, progress_bar
                    )
                
                # Display overall accuracy
                st.subheader("Evaluation Results")
                st.metric("Overall Accuracy", f"{overall_accuracy:.4f}")
                
                # Display class accuracies
                st.subheader("Class-wise Accuracy")
                
                # Create DataFrame for class accuracies
                class_acc_df = pd.DataFrame({
                    'Class': list(class_accuracies.keys()),
                    'Accuracy': list(class_accuracies.values())
                })
                class_acc_df = class_acc_df.sort_values('Accuracy', ascending=False)
                st.dataframe(class_acc_df)
                
                # Plot class accuracies
                class_acc_plot = plot_class_accuracies(class_accuracies)
                st.image(class_acc_plot)
                
                # Display confusion matrix
                st.subheader("Confusion Matrix")
                cm_plot = plot_confusion_matrix(cm, class_names)
                st.image(cm_plot)
                
                # Display detailed results
                st.subheader("Detailed Results")
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                
                # Download results as CSV
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="test_evaluation_results.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()