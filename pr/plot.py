import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, disk
from skimage.filters import threshold_otsu
import io

def visualize_tumor_detection(image):
    """
    Process the image and return multiple visualizations for tumor detection
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Create a dictionary to store all visualizations
    visualizations = {}
    
    # 1. Original Image
    visualizations['original'] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 2. Enhanced Image
    enhanced = cv2.equalizeHist(gray)
    visualizations['enhanced'] = cv2.cvtColor(cv2.applyColorMap(enhanced, cv2.COLORMAP_JET), 
                                            cv2.COLOR_BGR2RGB)
    
    # 3. Segmentation using Otsu's thresholding
    thresh = threshold_otsu(gray)
    binary = gray > thresh
    binary = clear_border(closing(binary, square(3)))
    visualizations['segmentation'] = binary
    
    # 4. Edge Detection
    edges = feature.canny(gray, sigma=2)
    visualizations['edges'] = edges
    
    # 5. Region Properties
    labeled_img = label(binary)
    regions = regionprops(labeled_img)
    
    # Create region overlay
    region_overlay = np.zeros_like(image)
    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        cv2.rectangle(region_overlay, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
    
    visualizations['regions'] = cv2.addWeighted(image, 0.7, region_overlay, 0.3, 0)
    
    # 6. Heatmap
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    visualizations['heatmap'] = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    return visualizations

def plot_visualizations(visualizations):
    """
    Create a figure with all visualizations
    """
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle('Tumor Detection Visualizations', fontsize=16)
    
    titles = {
        'original': 'Original Image',
        'enhanced': 'Enhanced Image',
        'segmentation': 'Segmentation',
        'edges': 'Edge Detection',
        'regions': 'Region Detection',
        'heatmap': 'Intensity Heatmap'
    }
    
    for idx, (key, image) in enumerate(visualizations.items(), 1):
        plt.subplot(2, 3, idx)
        plt.imshow(image, cmap='gray' if key in ['segmentation', 'edges'] else None)
        plt.title(titles[key])
        plt.axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    plt.close()
    
    return buf

def get_tumor_metrics(image):
    """
    Calculate tumor metrics from the image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Segment the image
    thresh = threshold_otsu(gray)
    binary = gray > thresh
    binary = clear_border(closing(binary, square(3)))
    
    # Get region properties
    labeled_img = label(binary)
    regions = regionprops(labeled_img)
    
    metrics = {
        'num_regions': len(regions),
        'regions': []
    }
    
    for region in regions:
        metrics['regions'].append({
            'area': region.area,
            'perimeter': region.perimeter,
            'eccentricity': region.eccentricity,
            'bbox': region.bbox
        })
    
    return metrics