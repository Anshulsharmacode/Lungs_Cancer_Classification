'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';


interface PredictionResponse {
  prediction: {
    predicted_class: string;
    confidence: number;
    probabilities: Record<string, number>;
  };
  visualizations: Record<string, string>;
  metrics: {
    tumor: {
      num_regions: number;
      statistics: {
        area: { mean: number; std: number };
        perimeter: { mean: number; std: number };
      };
    };
  };
  statistical_analysis: {
    intensity: {
      mean: number;
      std: number;
      skewness: number;
      kurtosis: number;
    };
    texture: Record<string, number>;
  };
  error: string | null;
}

interface BatchPredictionResponse {
  predictions: Array<{
    predicted_class: string;
    confidence: number;
    probabilities: Record<string, number>;
  }>;
  summary: {
    total_images: number;
    class_distribution: Record<string, { count: number; percentage: number }>;
  };
}

interface ImageModalProps {
  imageUrl: string;
  title: string;
  onClose: () => void;
}

// const colors = {
//   primary: {
//     light: '#EBF8FF',
//     medium: '#3B82F6',
//     dark: '#2563EB',
//   },
//   secondary: {
//     light: '#F0F9FF',
//     medium: '#0EA5E9',
//     dark: '#0284C7',
//   },
//   neutral: {
//     light: '#F8FAFC',
//     medium: '#64748B',
//     dark: '#1E293B',
//   },
//   success: {
//     light: '#F0FDF4',
//     medium: '#22C55E',
//   },
//   error: {
//     light: '#FEF2F2',
//     medium: '#EF4444',
//   }
// };

const ImageModal: React.FC<ImageModalProps> = ({ imageUrl, title, onClose }) => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4" onClick={onClose}>
      <div className="relative w-full max-w-5xl" onClick={e => e.stopPropagation()}>
        <div className="bg-white rounded-2xl shadow-2xl overflow-hidden border border-gray-100">
          <div className="p-3 sm:p-4 border-b border-gray-100 flex justify-between items-center bg-gradient-to-r from-primary-light to-secondary-light">
            <h3 className="text-lg sm:text-xl font-semibold text-neutral-dark truncate">{title}</h3>
            <button 
              onClick={onClose}
              className="p-1.5 sm:p-2 hover:bg-white/50 rounded-full transition-colors"
            >
              <svg className="w-5 h-5 sm:w-6 sm:h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div className="p-4 sm:p-6">
            <Image 
              src={imageUrl} 
              alt={title}
              width={800}
              height={600}
              className="w-full h-auto max-h-[70vh] sm:max-h-[80vh] object-contain rounded-lg"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

interface CardProps {
  children: React.ReactNode;
  className?: string;
}

const Card: React.FC<CardProps> = ({ children, className = '' }) => (
  <div className={`bg-white/95 backdrop-blur-sm rounded-2xl p-4 sm:p-6 shadow-lg border border-blue-100 hover:shadow-xl transition-all duration-300 ${className}`}>
    {children}
  </div>
);

interface MetricItemProps {
  label: string;
  value: string | number;
  unit?: string;
}

const MetricItem: React.FC<MetricItemProps> = ({ label, value, unit = '' }) => (
  <div className="flex justify-between items-center p-2 sm:p-3 rounded-lg bg-blue-50/50 hover:bg-blue-100/50 transition-colors">
    <span className="text-sm sm:text-base text-neutral-600">{label}</span>
    <span className="font-semibold text-blue-600 text-sm sm:text-base">
      {value}{unit}
    </span>
  </div>
);

export default function PredictPage() {
  const [isBatchMode, setIsBatchMode] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [batchPredictions, setBatchPredictions] = useState<PredictionResponse[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedImage, setSelectedImage] = useState<{ url: string; title: string } | null>(null);
  const [imageURLs, setImageURLs] = useState<string[]>([]);

  const simplifiedNames: { [key: string]: string } = {
    "Large.cell.carcinoma Left.hilum T2 N2 M0 IIIa": "Large Cell Carcinoma",
    "Squamous.cell.carcinoma Left.hilum T1 N2 M0 IIIa": "Squamous Cell Carcinoma", 
    "Adenocarcinoma Left.lower.lobe T2 N0 M0 Ib": "Adenocarcinoma",
    "Malignant cases": "Malignant",
    "Bengin cases": "Benign",
    "Normal": "Normal"
  };

  const formatClassName = (className: string | undefined) => {
    if (!className) return '';
    return simplifiedNames[className] || className;
  };

  const handleSinglePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('https://final-major-1.onrender.com/analyze', {
        method: 'POST',
        body: formData,
        credentials: 'include',
        headers: {
          'Accept': 'application/json',
        },
        mode: 'cors'
      });
      
      if (!response.ok) {
        throw new Error(`Prediction failed: ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Received data:', data);
      setPrediction(data);
    } catch (error) {
      setError('Error processing image. Please try again.');
      console.error('Error predicting:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleBatchPredict = async () => {
    if (!selectedFiles || selectedFiles.length === 0) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    Array.from(selectedFiles).forEach((file) => {
      formData.append('files', file);
    });

    try {
      const response = await fetch('https://final-major-1.onrender.com/analyze/batch', {
        method: 'POST',
        body: formData,
        credentials: 'include',
        headers: { 'Accept': 'application/json' },
        mode: 'cors'
      });

      if (!response.ok) {
        throw new Error(`Batch prediction failed: ${response.status}`);
      }

      const data: BatchPredictionResponse = await response.json();
      
      const transformedPredictions = data.predictions.map(pred => ({
        prediction: {
          predicted_class: pred.predicted_class,
          confidence: pred.confidence,
          probabilities: pred.probabilities
        },
        visualizations: {},
        metrics: {
          tumor: {
            num_regions: 0,
            statistics: {
              area: { mean: 0, std: 0 },
              perimeter: { mean: 0, std: 0 }
            }
          }
        },
        statistical_analysis: {
          intensity: {
            mean: 0,
            std: 0,
            skewness: 0,
            kurtosis: 0
          },
          texture: {}
        },
        error: null
      }));

      setBatchPredictions(transformedPredictions);
      console.log('Batch Summary:', data.summary);
    } catch (error) {
      setError('Error processing batch. Please try again.');
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (files: FileList | null) => {
    if (isBatchMode) {
      setSelectedFiles(files);
      const urls = files ? Array.from(files).map(file => URL.createObjectURL(file)) : [];
      setImageURLs(urls);
    } else {
      setSelectedFile(files?.[0] || null);
      setImageURLs(files?.[0] ? [URL.createObjectURL(files[0])] : []);
    }
  };

  useEffect(() => {
    return () => {
      imageURLs.forEach(url => URL.revokeObjectURL(url));
    };
  }, [imageURLs]);

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-blue-100">
      <div className="flex flex-col lg:flex-row h-auto w-auto">
        {/* Left Panel - File Upload & Controls */}
        <div className="w-full lg:w-1/3 xl:w-1/4 h-auto lg:h-full border-b lg:border-b-0 lg:border-r border-blue-100 overflow-y-auto p-4">
          <div className="space-y-4 sm:space-y-6">
            <div className="space-y-2">
              <h1 className="text-2xl sm:text-3xl font-bold text-blue-900">
                Lung Cancer Detection
              </h1>
              <p className="text-base sm:text-lg text-blue-600">
                Advanced Image Analysis System
              </p>
            </div>

            {/* Mode Selection */}
            <div className="flex gap-2 sm:gap-4">
              <button
                onClick={() => {
                  setIsBatchMode(false);
                  setSelectedFiles(null);
                  setBatchPredictions([]);
                }}
                className={`flex-1 px-3 sm:px-4 py-2 sm:py-3 rounded-lg text-sm sm:text-base transition-all duration-300 ${
                  !isBatchMode 
                    ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-lg'
                    : 'bg-white text-blue-600 hover:bg-blue-50 border border-blue-200'
                }`}
              >
                Single Image
              </button>
              <button
                onClick={() => {
                  setIsBatchMode(true);
                  setSelectedFile(null);
                  setPrediction(null);
                }}
                className={`flex-1 px-3 sm:px-4 py-2 sm:py-3 rounded-lg text-sm sm:text-base transition-all duration-300 ${
                  isBatchMode 
                    ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-lg'
                    : 'bg-white text-blue-600 hover:bg-blue-50 border border-blue-200'
                }`}
              >
                Batch Analysis
              </button>
            </div>

            {/* File Upload Card */}
            <Card>
              <label className="block text-base sm:text-lg font-semibold text-blue-900 mb-4">
                Upload {isBatchMode ? 'Multiple CT Scan Images' : 'CT Scan Image'}
              </label>
              <input
                type="file"
                onChange={(e) => handleFileSelect(e.target.files)}
                accept="image/*"
                multiple={isBatchMode}
                className="block w-full text-xs sm:text-sm text-blue-600 file:mr-4 file:py-2 file:px-4 
                         sm:file:py-3 sm:file:px-6 file:rounded-full file:border-0 
                         file:text-sm file:font-semibold file:bg-gradient-to-r 
                         file:from-blue-500 file:to-blue-600 file:text-white 
                         hover:file:bg-blue-600 file:transition-colors cursor-pointer"
              />
              
              {/* Preview Grid */}
              {imageURLs.length > 0 && (
                <div className="mt-4 sm:mt-6 grid grid-cols-2 gap-2 sm:gap-3">
                  {imageURLs.map((url, index) => (
                    <div 
                      key={index}
                      className="relative aspect-square rounded-xl overflow-hidden cursor-pointer 
                               hover:ring-2 hover:ring-blue-400 transition-all duration-300"
                      onClick={() => setSelectedImage({ url, title: `Original Image ${index + 1}` })}
                    >
                      <Image 
                        src={url} 
                        alt={`Preview ${index + 1}`}
                        width={800}
                        height={600}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                      />
                      <div className="bg-blue-900/80 py-1 px-2 absolute bottom-0 w-full">
                        <p className="text-white text-xs sm:text-sm font-medium text-center">
                          Image {index + 1}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Analyze Button */}
              <button
                onClick={isBatchMode ? handleBatchPredict : handleSinglePredict}
                disabled={isBatchMode ? !selectedFiles?.length : !selectedFile || loading}
                className="w-full mt-4 sm:mt-6 px-4 sm:px-6 py-2 sm:py-3 rounded-lg 
                         text-sm sm:text-base transition-all duration-300 
                         bg-gradient-to-r from-blue-500 to-blue-600 text-white
                         hover:from-blue-600 hover:to-blue-700 shadow-md hover:shadow-lg
                         disabled:from-gray-200 disabled:to-gray-300 
                         disabled:text-gray-400 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-4 w-4 sm:h-5 sm:w-5" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"/>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"/>
                    </svg>
                    Processing...
                  </span>
                ) : (
                  `Analyze ${isBatchMode ? 'Images' : 'Image'}`
                )}
              </button>
            </Card>

            {/* Error Display */}
            {error && (
              <div className="p-3 sm:p-4 bg-red-50 border border-red-200 rounded-xl text-red-600 text-sm sm:text-base">
                {error}
              </div>
            )}
          </div>
        </div>

        {/* Right Panel - Analysis Results */}
        <div className="flex-1 h-auto lg:h-full overflow-y-auto">
          {!isBatchMode && prediction && prediction.prediction && (
            <div className="h-full flex flex-col">
              {/* Analysis Results Header */}
              <div className="sticky top-0 z-10 bg-white/95 backdrop-blur-sm border-b border-blue-100 p-4 sm:p-6">
                <div className="flex items-center justify-between">
                  <h3 className="text-lg sm:text-xl font-bold text-blue-900">Analysis Results</h3>
                  <span className="text-xs sm:text-sm bg-blue-100 text-blue-800 px-2 sm:px-3 py-1 rounded-full">
                    {(prediction.prediction.predicted_class)}
                    ({(prediction.prediction.confidence * 100).toFixed(1)}%)
                  </span>
                </div>
              </div>

              <div className="flex-1 p-4 sm:p-6 space-y-4 sm:space-y-6">
                {/* Probability Distribution */}
                <Card>
                  <h4 className="font-semibold text-blue-900 mb-4">Probability Distribution</h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    {Object.entries(prediction.prediction.probabilities)
                      .sort(([, a], [, b]) => b - a)
                      .map(([key, value]) => (
                        <div key={key} className="relative">
                          <div className="flex flex-col space-y-2">
                            <div className="flex justify-between items-center">
                              <span className="text-xs sm:text-sm font-medium text-blue-900">{formatClassName(key)}</span>
                              <span className="text-xs sm:text-sm font-bold text-blue-700">
                                {(value * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="h-2 sm:h-3 w-full bg-blue-100 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-300"
                                style={{ width: `${value * 100}%` }}
                              />
                            </div>
                          </div>
                        </div>
                      ))}
                  </div>
                </Card>

                {/* Main CT Scan Display */}
                {selectedImage && (
                  <Card>
                    <div className="aspect-[16/9] w-full rounded-xl overflow-hidden">
                      <Image 
                        src={selectedImage.url}
                        alt={selectedImage.title}
                        width={800}
                        height={600}
                        className="w-full h-full object-contain"
                      />
                    </div>
                  </Card>
                )}

                {/* Visualization Grid */}
                {prediction.visualizations && Object.keys(prediction.visualizations).length > 0 && (
                  <Card>
                    <h4 className="font-semibold text-blue-900 mb-4">Analysis Visualizations</h4>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      {Object.entries(prediction.visualizations).map(([key, base64Data]) => (
                        <div 
                          key={key}
                          className="aspect-[4/3] rounded-xl overflow-hidden cursor-pointer 
                                   hover:ring-2 hover:ring-blue-400 transition-all duration-300"
                          onClick={() => setSelectedImage({
                            url: `data:image/png;base64,${base64Data}`,
                            title: formatClassName(key)
                          })}
                        >
                          <Image 
                            src={`data:image/png;base64,${base64Data}`}
                            alt={formatClassName(key)}
                            width={800}
                            height={600}
                            className="w-full h-full object-cover"
                          />
                          <div className="bg-blue-900/80 py-1 sm:py-2 px-2 sm:px-3">
                            <p className="text-white text-xs sm:text-sm font-medium text-center">
                              {formatClassName(key)}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </Card>
                )}

                {/* Metrics & Statistics */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                  {/* Tumor Metrics */}
                  {prediction.metrics && (
                    <Card>
                      <h4 className="font-semibold text-blue-900 mb-3">Tumor Metrics</h4>
                      <div className="space-y-2">
                        <MetricItem 
                          label="Number of Regions"
                          value={prediction.metrics.tumor.num_regions}
                        />
                        <MetricItem 
                          label="Average Area"
                          value={prediction.metrics.tumor.statistics.area.mean.toFixed(2)}
                        />
                        <MetricItem 
                          label="Average Perimeter"
                          value={prediction.metrics.tumor.statistics.perimeter.mean.toFixed(2)}
                        />
                        <MetricItem 
                          label="Area Std Dev"
                          value={prediction.metrics.tumor.statistics.area.std.toFixed(2)}
                        />
                        <MetricItem 
                          label="Perimeter Std Dev"
                          value={prediction.metrics.tumor.statistics.perimeter.std.toFixed(2)}
                        />
                      </div>
                    </Card>
                  )}

                  {/* Image Statistics */}
                  {prediction.statistical_analysis && (
                    <Card>
                      <h4 className="font-semibold text-blue-900 mb-3">Image Statistics</h4>
                      <div className="space-y-2">
                        <MetricItem 
                          label="Mean Intensity"
                          value={prediction.statistical_analysis.intensity.mean.toFixed(2)}
                        />
                        <MetricItem 
                          label="Standard Deviation"
                          value={prediction.statistical_analysis.intensity.std.toFixed(2)}
                        />
                        <MetricItem 
                          label="Skewness"
                          value={prediction.statistical_analysis.intensity.skewness.toFixed(2)}
                        />
                        <MetricItem 
                          label="Kurtosis"
                          value={prediction.statistical_analysis.intensity.kurtosis.toFixed(2)}
                        />
                      </div>
                    </Card>
                  )}

                  {/* Texture Analysis */}
                  {prediction.statistical_analysis?.texture && (
                    <Card>
                      <h4 className="font-semibold text-blue-900 mb-3">Texture Analysis</h4>
                      <div className="h-[180px] overflow-y-auto pr-2 space-y-2">
                        {Object.entries(prediction.statistical_analysis.texture).map(([key, value]) => (
                          <MetricItem 
                            key={key}
                            label={formatClassName(key)}
                            value={typeof value === 'number' ? value.toFixed(5) : value}
                          />
                        ))}
                      </div>
                    </Card>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Batch Mode Results */}
          {isBatchMode && batchPredictions.length > 0 && (
            <Card>
              <h3 className="text-xl font-bold text-blue-900 mb-6">Batch Analysis Results</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {batchPredictions.map((pred, index) => (
                  <div key={index} className="grid grid-cols-1 gap-3 p-4 bg-blue-50 rounded-xl border border-blue-200">
                    {imageURLs[index] && (
                      <div 
                        className="aspect-square rounded-xl overflow-hidden cursor-pointer"
                        onClick={() => setSelectedImage({ url: imageURLs[index], title: `Image ${index + 1}` })}
                      >
                        <Image 
                          src={imageURLs[index]} 
                          alt={`Original ${index + 1}`}
                          width={800}
                          height={600}
                          className="w-full h-full object-cover hover:scale-105 transition-transform duration-300"
                        />
                        <div className="bg-blue-900/80 py-1 px-2 absolute bottom-0 w-full">
                          <p className="text-white text-xs sm:text-sm font-medium text-center">
                            Image {index + 1}
                          </p>
                        </div>
                      </div>
                    )}
                    <div className="p-3 sm:p-4 bg-white rounded-lg">
                      <h4 className="font-semibold text-blue-900 mb-3">Prediction Results</h4>
                      {pred.prediction.probabilities && (
                        <div className="space-y-2 sm:space-y-3">
                          {Object.entries(pred.prediction.probabilities)
                            .sort(([, a], [, b]) => b - a)
                            .map(([key, value]) => (
                              <div key={key} className="relative">
                                <div className="flex items-center gap-2">
                                  <div className="w-1/3 text-xs sm:text-sm text-blue-800">
                                    {formatClassName(key)}
                                  </div>
                                  <div className="flex-1 h-5 sm:h-6 relative">
                                    <div className="absolute inset-y-0 w-full bg-blue-50 rounded-full" />
                                    <div 
                                      className="absolute inset-y-0 rounded-full bg-blue-500"
                                      style={{ width: `${value * 100}%` }}
                                    />
                                    <span className={`absolute inset-y-0 right-2 flex items-center text-xs font-medium ${
                                      value > 0.4 ? 'text-white' : 'text-blue-800'
                                    }`}>
                                      {(value * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                </div>
                              </div>
                            ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>
      </div>

      {/* Image Modal */}
      {selectedImage && (
        <ImageModal 
          imageUrl={selectedImage.url}
          title={selectedImage.title}
          onClose={() => setSelectedImage(null)}
        />
      )}
    </main>
  );
}