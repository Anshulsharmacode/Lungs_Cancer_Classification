from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import pickle
import os
from typing import List
from pydantic import BaseModel
import io
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops, hog
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
from plot import visualize_tumor_detection, plot_visualizations, get_tumor_metrics
# Add this import and set backend at the very top of the file, before other imports
import matplotlib
matplotlib.use('Agg')

# Initialize FastAPI app
app = FastAPI(
    title="Lung Cancer Classification API",
    description="API for lung cancer classification using computer vision",
    version="1.0.0"
)

# Add CORS middleware with specific frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Define paths
DATA_DIR = "data"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "cv_ensemble_model.pkl")
TEST_DIR = os.path.join(DATA_DIR, "Data/test")

# Load model at startup
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data['ensemble'], model_data['scaler'], model_data['class_names']
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")

# Load model globally
try:
    model, scaler, class_names = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at startup: {e}")

# Feature extraction function (same as in UI.py)
def extract_features(img, img_size=224):
    """Extract features from an image with robust error handling"""
    try:
        if not isinstance(img, np.ndarray):
            raise ValueError("Input must be a numpy array")
        
        if img.size == 0:
            raise ValueError("Empty image")
            
        # Handle different color spaces
        if len(img.shape) == 2:  # Grayscale
            gray = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif len(img.shape) == 3:
            if img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif img.shape[2] == 3:  # BGR/RGB
                pass
            else:
                raise ValueError(f"Unsupported number of channels: {img.shape[2]}")
        else:
            raise ValueError(f"Invalid image dimensions: {img.shape}")
            
        # Resize image
        try:
            img = cv2.resize(img, (img_size, img_size))
        except Exception as e:
            raise ValueError(f"Failed to resize image: {str(e)}")
            
        # Convert to grayscale for feature extraction
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            raise ValueError(f"Failed to convert image to grayscale: {str(e)}")
        
        features = []
        
        # 1. Basic statistical features
        try:
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            mean = np.mean(gray)
            std = np.std(gray)
            skewness = np.mean(((gray - mean)/std)**3) if std > 0 else 0
            kurtosis = np.mean(((gray - mean)/std)**4) if std > 0 else 0
            
            features.extend([mean, std, skewness, kurtosis])
        except Exception as e:
            raise ValueError(f"Failed to extract statistical features: {str(e)}")
        
        # 2. Texture features (GLCM)
        try:
            glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                               symmetric=True, normed=True)
            
            glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            for prop in glcm_props:
                features.extend(graycoprops(glcm, prop).flatten())
        except Exception as e:
            raise ValueError(f"Failed to extract GLCM features: {str(e)}")
        
        # 3. Local Binary Patterns
        try:
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp, bins=n_points+2, range=(0, n_points+2), density=True)
            features.extend(lbp_hist)
        except Exception as e:
            raise ValueError(f"Failed to extract LBP features: {str(e)}")
        
        # 4. HOG features
        try:
            hog_features, _ = hog(gray, orientations=9, pixels_per_cell=(16, 16),
                               cells_per_block=(2, 2), visualize=True, feature_vector=True)
            hog_features_subset = hog_features[::10]  # Take every 10th feature to reduce dimensionality
            features.extend(hog_features_subset)
        except Exception as e:
            raise ValueError(f"Failed to extract HOG features: {str(e)}")
        
        # 5. Shape and edge features
        try:
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (img_size * img_size)
            features.append(edge_density)
            
            # Fourier features
            f_transform = np.fft.fft2(gray)
            f_transform_shifted = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
            mag_mean = np.mean(magnitude_spectrum)
            mag_std = np.std(magnitude_spectrum)
            features.extend([mag_mean, mag_std])
        except Exception as e:
            raise ValueError(f"Failed to extract shape/edge features: {str(e)}")
        
        return np.array(features)
        
    except Exception as e:
        raise ValueError(f"Feature extraction failed: {str(e)}")

# Prediction function
def predict_image(image):
    try:
        # Extract features
        features = extract_features(image)
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Get predicted class and confidence
        predicted_class = class_names[prediction]
        confidence = float(probabilities[prediction])  # Convert to float for JSON serialization
        
        # Create probability dictionary
        all_probs = {class_name: float(prob) for class_name, prob in zip(class_names, probabilities)}
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": all_probs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pydantic models for response
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: dict

# Add new Pydantic models for visualization responses
class TumorMetricsResponse(BaseModel):
    num_regions: int
    avg_area: float = None
    avg_perimeter: float = None
    avg_eccentricity: float = None

class VisualizationResponse(BaseModel):
    plot_base64: str
    metrics: TumorMetricsResponse

def encode_plot_to_base64(fig):
    """Helper function to encode matplotlib figure to base64"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return plot_base64

@app.post("/visualize")
async def get_visualizations(file: UploadFile = File(...)):
    """Generate visualizations using plot.py functions"""
    try:
        # Read and validate image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        response = {
            "visualizations": {},
            "metrics": None,
            "error": None
        }

        # 1. Get tumor visualizations using plot.py functions
        try:
            # Get all visualizations from plot.py
            tumor_viz = visualize_tumor_detection(image)
            
            # Encode each visualization
            for key, img in tumor_viz.items():
                plt.figure(figsize=(8, 8))
                if isinstance(img, np.ndarray):
                    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                        plt.imshow(img, cmap='gray')
                    else:
                        plt.imshow(img)
                plt.title(key.replace('_', ' ').title())
                plt.axis('off')
                plt.tight_layout()
                response["visualizations"][key] = encode_plot_to_base64(plt.gcf())
                plt.close('all')

            # Get combined visualization plot
            combined_viz = plot_visualizations(tumor_viz)
            response["visualizations"]["combined"] = base64.b64encode(combined_viz.getvalue()).decode()
            
        except Exception as e:
            response["error"] = f"Visualization error: {str(e)}"

        # 2. Get tumor metrics from plot.py
        try:
            metrics = get_tumor_metrics(image)
            response["metrics"] = {
                "num_regions": metrics['num_regions'],
                "regions": [{
                    "area": region['area'],
                    "perimeter": region['perimeter'],
                    "eccentricity": region['eccentricity']
                } for region in metrics['regions']],
                "statistics": {
                    "area": {
                        "mean": float(np.mean([r['area'] for r in metrics['regions']])) if metrics['regions'] else 0,
                        "std": float(np.std([r['area'] for r in metrics['regions']])) if metrics['regions'] else 0
                    },
                    "perimeter": {
                        "mean": float(np.mean([r['perimeter'] for r in metrics['regions']])) if metrics['regions'] else 0,
                        "std": float(np.std([r['perimeter'] for r in metrics['regions']])) if metrics['regions'] else 0
                    },
                    "eccentricity": {
                        "mean": float(np.mean([r['eccentricity'] for r in metrics['regions']])) if metrics['regions'] else 0,
                        "std": float(np.std([r['eccentricity'] for r in metrics['regions']])) if metrics['regions'] else 0
                    }
                }
            }
        except Exception as e:
            response["error"] = f"{response.get('error', '')} Metrics error: {str(e)}"

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Remove other visualization routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Lung Cancer Classification API",
        "version": "1.0.0",
        "model_status": "loaded" if model is not None else "not loaded",
        "available_classes": class_names
    }

@app.post("/predict", response_model=None)  # Remove response_model to allow custom response
async def predict_single(file: UploadFile = File(...)):
    """Predict a single image with visualizations"""
    try:
        # Read and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Get prediction
        prediction_result = predict_image(image)
        
        # Get visualizations
        response = {
            "prediction": prediction_result,
            "plots": {},
            "error": None
        }

        try:
            # Get tumor visualizations
            tumor_viz = visualize_tumor_detection(image)
            
            # Encode each visualization
            for key, img in tumor_viz.items():
                plt.figure(figsize=(8, 8))
                if isinstance(img, np.ndarray):
                    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                        plt.imshow(img, cmap='gray')
                    else:
                        plt.imshow(img)
                plt.title(key.replace('_', ' ').title())
                plt.axis('off')
                plt.tight_layout()
                response["plots"][key] = encode_plot_to_base64(plt.gcf())
                plt.close('all')

            # Add probability visualization
            plt.figure(figsize=(10, 6))
            probs = prediction_result["probabilities"]
            plt.bar(probs.keys(), probs.values(), color='skyblue')
            plt.title('Class Probabilities')
            plt.xlabel('Classes')
            plt.ylabel('Probability')
            plt.xticks(rotation=45)
            plt.tight_layout()
            response["plots"]["probability_distribution"] = encode_plot_to_base64(plt.gcf())
            plt.close('all')

        except Exception as e:
            response["error"] = f"Visualization error: {str(e)}"

        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """Predict multiple images"""
    try:
        predictions = []
        class_counts = {class_name: 0 for class_name in class_names}
        
        for file in files:
            # Read and decode image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                continue
            
            # Make prediction
            result = predict_image(image)
            predictions.append(result)
            class_counts[result["predicted_class"]] += 1
        
        # Calculate summary
        total = len(predictions)
        summary = {
            "total_images": total,
            "class_distribution": {
                class_name: {
                    "count": count,
                    "percentage": count/total if total > 0 else 0
                }
                for class_name, count in class_counts.items()
            }
        }
        
        return {
            "predictions": predictions,
            "summary": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_type": str(type(model).__name__),
        "number_of_classes": len(class_names),
        "classes": class_names,
        "model_path": MODEL_PATH
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)