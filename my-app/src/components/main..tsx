'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

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

const colors = {
  primary: {
    light: '#EBF8FF',
    medium: '#3B82F6',
    dark: '#2563EB',
  },
  secondary: {
    light: '#F0F9FF',
    medium: '#0EA5E9',
    dark: '#0284C7',
  },
  neutral: {
    light: '#F8FAFC',
    medium: '#64748B',
    dark: '#1E293B',
  },
  success: {
    light: '#F0FDF4',
    medium: '#22C55E',
  },
  error: {
    light: '#FEF2F2',
    medium: '#EF4444',
  }
};

const ImageModal: React.FC<ImageModalProps> = ({ imageUrl, title, onClose }) => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm" onClick={onClose}>
      <div className="relative max-w-5xl w-full mx-4" onClick={e => e.stopPropagation()}>
        <div className="bg-white rounded-2xl shadow-2xl overflow-hidden border border-gray-100">
          <div className="p-4 border-b border-gray-100 flex justify-between items-center bg-gradient-to-r from-primary-light to-secondary-light">
            <h3 className="text-xl font-semibold text-neutral-dark">{title}</h3>
            <button 
              onClick={onClose}
              className="p-2 hover:bg-white/50 rounded-full transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div className="p-6">
            <img 
              src={imageUrl} 
              alt={title}
              className="w-full h-auto max-h-[80vh] object-contain rounded-lg"
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
  <div className={`bg-white/95 backdrop-blur-sm rounded-2xl p-6 shadow-lg border border-blue-100 hover:shadow-xl transition-all duration-300 ${className}`}>
    {children}
  </div>
);

interface MetricItemProps {
  label: string;
  value: string | number;
  unit?: string;
}

const MetricItem: React.FC<MetricItemProps> = ({ label, value, unit = '' }) => (
  <div className="flex justify-between items-center p-3 rounded-lg bg-blue-50/50 hover:bg-blue-100/50 transition-colors">
    <span className="text-neutral-600">{label}</span>
    <span className="font-semibold text-blue-600">
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

  const formatClassName = (className: string | undefined) => {
    if (!className) return '';
    try {
      return className
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
    } catch (error) {
      console.error('Error formatting class name:', error);
      return className || '';
    }
  };

  const handleSinglePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('http://localhost:8000/analyze', {
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
      const response = await fetch('http://localhost:8000/analyze/batch', {
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
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-blue-100 relative overflow-x-hidden">
      <div className="absolute -right-40 -top-40 w-96 h-96 bg-blue-200 opacity-10 rounded-full blur-3xl" />
      <div className="absolute -left-40 -bottom-40 w-96 h-96 bg-blue-200 opacity-10 rounded-full blur-3xl" />
      
      <div className="container mx-auto px-6 py-16 relative">
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-12 gap-6">
          <div className="space-y-2">
            <h1 className="text-4xl font-bold text-blue-900 leading-tight">
              Lung Cancer Detection
            </h1>
            <p className="text-xl text-blue-600">
              Advanced Image Analysis System
            </p>
          </div>
          
          <div className="flex gap-4">
            <button
              onClick={() => {
                setIsBatchMode(false);
                setSelectedFiles(null);
                setBatchPredictions([]);
              }}
              className={`px-6 py-3 rounded-lg transition-all duration-300 ${
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
              className={`px-6 py-3 rounded-lg transition-all duration-300 ${
                isBatchMode 
                  ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-lg'
                  : 'bg-white text-blue-600 hover:bg-blue-50 border border-blue-200'
              }`}
            >
              Batch Analysis
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
          <div className="xl:col-span-5 space-y-6">
            <Card>
              <label className="block text-lg font-semibold text-blue-900 mb-4">
                Upload {isBatchMode ? 'Multiple CT Scan Images' : 'CT Scan Image'}
              </label>
              <input
                type="file"
                onChange={(e) => handleFileSelect(e.target.files)}
                accept="image/*"
                multiple={isBatchMode}
                className="block w-full text-sm text-blue-600 file:mr-4 file:py-3 file:px-6 
                         file:rounded-full file:border-0 file:text-sm file:font-semibold 
                         file:bg-gradient-to-r file:from-blue-500 file:to-blue-600 
                         file:text-white hover:file:bg-blue-600
                         file:transition-colors cursor-pointer"
              />
              
              {imageURLs.length > 0 && (
                <div className="mt-6 grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                  {imageURLs.map((url, index) => (
                    <div 
                      key={index}
                      className="relative aspect-square rounded-xl overflow-hidden cursor-pointer 
                               group hover:ring-2 hover:ring-blue-400 transition-all duration-300"
                      onClick={() => setSelectedImage({ url, title: `Original Image ${index + 1}` })}
                    >
                      <img 
                        src={url} 
                        alt={`Preview ${index + 1}`}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                        <p className="absolute bottom-2 left-2 text-white text-sm font-medium">
                          Image {index + 1}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              <button
                onClick={isBatchMode ? handleBatchPredict : handleSinglePredict}
                disabled={isBatchMode ? !selectedFiles?.length : !selectedFile || loading}
                className="w-full mt-6 px-6 py-3 rounded-lg transition-all duration-300 
                         bg-gradient-to-r from-blue-500 to-blue-600 text-white
                         hover:from-blue-600 hover:to-blue-700 shadow-md hover:shadow-lg
                         disabled:from-gray-200 disabled:to-gray-300 
                         disabled:text-gray-400 disabled:cursor-not-allowed"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
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

            {error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-red-600">
                {error}
              </div>
            )}

            {!isBatchMode && prediction && prediction.prediction && (
              <Card>
                <h3 className="text-xl font-bold text-blue-900 mb-6">Analysis Results</h3>
                <div className="space-y-4">
                  <div className="p-5 bg-gradient-to-r from-blue-50 to-blue-100 rounded-xl border border-blue-200">
                    <div className="space-y-4">
                      <div className="flex justify-between items-center p-3 bg-white/80 rounded-lg">
                        <span className="text-blue-800">Predicted Class</span>
                        <span className="font-semibold text-blue-600 text-lg">
                          {formatClassName(prediction.prediction.predicted_class)}
                        </span>
                      </div>
                      <div className="flex justify-between items-center p-3 bg-white/80 rounded-lg">
                        <span className="text-blue-800">Confidence Level</span>
                        <span className="font-semibold text-blue-600 text-lg">
                          {(prediction.prediction.confidence * 100).toFixed(2)}%
                        </span>
                      </div>
                    </div>
                  </div>

                  {prediction.prediction.probabilities && (
                    <div className="p-5 bg-white rounded-xl border border-blue-200">
                      <h4 className="font-semibold text-blue-900 mb-4">Probability Distribution</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4 gap-y-3">
                        {Object.entries(prediction.prediction.probabilities)
                          .sort(([, a], [, b]) => b - a)
                          .map(([key, value]) => (
                            <div key={key} className="relative">
                              <div className="flex flex-col h-full">
                                <div className="text-sm text-blue-800 font-medium mb-1">
                                  {formatClassName(key)}
                                </div>
                                <div className="flex-1 h-7 relative">
                                  <div className="absolute inset-y-0 w-full bg-blue-50 rounded-lg" />
                                  <div 
                                    className="absolute inset-y-0 rounded-lg transition-all duration-500"
                                    style={{
                                      width: `${value * 100}%`,
                                      background: value > 0.5 
                                        ? 'linear-gradient(90deg, #3B82F6 0%, #2563EB 100%)'
                                        : value > 0.25
                                        ? 'linear-gradient(90deg, #60A5FA 0%, #3B82F6 100%)'
                                        : 'linear-gradient(90deg, #93C5FD 0%, #60A5FA 100%)'
                                    }}
                                  />
                                  <div className="absolute inset-0 flex items-center justify-end pr-3">
                                    <span className={`text-xs font-bold ${
                                      value > 0.4 ? 'text-white' : 'text-blue-800'
                                    }`}>
                                      {(value * 100).toFixed(1)}%
                                    </span>
                                  </div>
                                </div>
                              </div>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                </div>
              </Card>
            )}
          </div>

          <div className="xl:col-span-7 space-y-6">
            {isBatchMode && batchPredictions.length > 0 && (
              <Card>
                <h3 className="text-xl font-bold text-blue-900 mb-6">Batch Analysis Results</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {batchPredictions.map((pred, index) => (
                    <div key={index} className="grid grid-cols-1 gap-3 p-4 bg-gradient-to-r from-blue-50 to-blue-100 rounded-xl border border-blue-200">
                      {imageURLs[index] && (
                        <div 
                          className="aspect-square rounded-xl overflow-hidden cursor-pointer group"
                          onClick={() => setSelectedImage({ url: imageURLs[index], title: `Image ${index + 1}` })}
                        >
                          <img 
                            src={imageURLs[index]} 
                            alt={`Original ${index + 1}`}
                            className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                          />
                        </div>
                      )}
                        <div className="p-4 bg-white/90 rounded-lg">
                          <h4 className="font-semibold text-blue-900 mb-3">Prediction Results</h4>
                          {pred.prediction.probabilities && (
                            <div className="space-y-3">
                              {Object.entries(pred.prediction.probabilities)
                                .sort(([, a], [, b]) => b - a)
                                .map(([key, value]) => (
                                  <div key={key} className="relative">
                                    <div className="flex items-center gap-2">
                                      <div className="w-1/3 text-sm text-blue-800">
                                        {formatClassName(key)}
                                      </div>
                                      <div className="flex-1 h-6 relative">
                                        <div className="absolute inset-y-0 w-full bg-blue-50 rounded-full" />
                                        <div 
                                          className="absolute inset-y-0 rounded-full transition-all duration-500"
                                          style={{
                                            width: `${value * 100}%`,
                                            background: value > 0.5 
                                              ? 'linear-gradient(90deg, #3B82F6 0%, #2563EB 100%)'
                                              : value > 0.25
                                              ? 'linear-gradient(90deg, #60A5FA 0%, #3B82F6 100%)'
                                              : 'linear-gradient(90deg, #93C5FD 0%, #60A5FA 100%)'
                                          }}
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

            {!isBatchMode && prediction && (
              <Card>
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-bold text-blue-900">Analysis Details</h3>
                  <span className="text-sm bg-blue-100 text-blue-800 px-3 py-1 rounded-full">
                          {formatClassName(prediction.prediction.predicted_class)}
                    ({(prediction.prediction.confidence * 100).toFixed(1)}%)
                        </span>
                  </div>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                  <div className="lg:col-span-7 space-y-6">
                    <h4 className="font-semibold text-blue-900">Visualizations</h4>
                    {prediction.visualizations && Object.keys(prediction.visualizations).length > 0 ? (
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                        {Object.entries(prediction.visualizations).map(([key, base64Data]) => (
                    <div 
                      key={key}
                      className="group relative aspect-square rounded-xl overflow-hidden cursor-pointer 
                               hover:ring-2 hover:ring-blue-400 transition-all duration-300"
                      onClick={() => setSelectedImage({
                        url: `data:image/png;base64,${base64Data}`,
                        title: formatClassName(key)
                      })}
                    >
                      <img 
                        src={`data:image/png;base64,${base64Data}`}
                        alt={formatClassName(key)}
                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity">
                              <div className="absolute bottom-0 left-0 right-0 p-3">
                                <p className="text-white font-medium text-sm">{formatClassName(key)}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                      </div>
                    ) : (
                      <div className="p-4 bg-blue-50 rounded-xl text-blue-700 text-center">
                        No visualizations available
                      </div>
                    )}

                    <div className="p-3 bg-blue-50 rounded-xl border border-blue-200">
                      <p className="text-sm text-blue-700">
                        Click on any visualization for a detailed view
                      </p>
                    </div>
                </div>

                  <div className="lg:col-span-5 space-y-6">
                    <h4 className="font-semibold text-blue-900">Metrics & Statistics</h4>
                    <div className="grid grid-cols-1 gap-4">
                {prediction.metrics && (
                        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-4 border border-blue-200">
                          <h5 className="text-sm font-semibold text-blue-900 mb-3">Tumor Metrics</h5>
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
                    </div>
                      )}

                    {prediction.statistical_analysis && (
                        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-4 border border-blue-200">
                          <h5 className="text-sm font-semibold text-blue-900 mb-3">Image Statistics</h5>
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
                      </div>
                    )}

                    {prediction.statistical_analysis?.texture && (
                        <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-4 border border-blue-200">
                          <h5 className="text-sm font-semibold text-blue-900 mb-3">Texture Analysis</h5>
                          <div className="h-[180px] overflow-y-auto pr-2 space-y-2">
                          {Object.entries(prediction.statistical_analysis.texture).map(([key, value]) => (
                            <MetricItem 
                              key={key}
                              label={formatClassName(key)}
                              value={typeof value === 'number' ? value.toFixed(3) : value}
                            />
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                  </div>
                </div>
              </Card>
            )}
        </div>
      </div>

      {selectedImage && (
        <ImageModal 
          imageUrl={selectedImage.url}
          title={selectedImage.title}
          onClose={() => setSelectedImage(null)}
        />
      )}
      </div>
    </main>
  );
}