'use client';

import { useState } from 'react';
import Link from 'next/link';

interface PredictionResponse {
  prediction: {
    predicted_class: string;
    confidence: number;
    probabilities: Record<string, number>;
  };
  plots: Record<string, string>;
  error: string | null;
}

interface BatchPredictionResponse {
  predictions: {
    predicted_class: string;
    confidence: number;
    probabilities: Record<string, number>;
  }[];
  summary: {
    total_images: number;
    class_distribution: Record<string, { count: number; percentage: number }>;
  };
}

export default function PredictPage() {
  const [isBatchMode, setIsBatchMode] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<FileList | null>(null);
  const [batchPredictions, setBatchPredictions] = useState<PredictionResponse[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
      const response = await fetch('http://localhost:8000/predict', {
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
      console.log('Received data:', data); // Debug log
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
      const response = await fetch('http://localhost:8000/predict/batch', {
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
      // Transform the data to match the expected format
      const transformedPredictions = data.predictions.map(pred => ({
        prediction: {
          predicted_class: pred.predicted_class,
          confidence: pred.confidence,
          probabilities: pred.probabilities
        },
        plots: {} // Add empty plots since batch doesn't return plots
      }));
      
      setBatchPredictions(transformedPredictions.map(pred => ({
        ...pred,
        error: null
      })));
    } catch (error) {
      setError('Error processing batch. Please try again.');
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 relative overflow-hidden">
      <div className="container mx-auto px-6 py-16 relative">
        <div className="flex items-center justify-between mb-12">
          <h1 className="text-4xl font-bold text-gray-800 leading-tight">
            Lung Cancer Detection
            <span className="block mt-2 text-2xl text-gray-600 font-medium">Image Analysis</span>
          </h1>
          <div className="flex gap-4">
            <button
              onClick={() => {
                setIsBatchMode(false);
                setSelectedFiles(null);
                setBatchPredictions([]);
              }}
              className={`px-6 py-3 rounded-lg transition-all duration-300 ${
                !isBatchMode 
                  ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white'
                  : 'bg-gray-100 text-gray-600'
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
                  ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white'
                  : 'bg-gray-100 text-gray-600'
              }`}
            >
              Batch Analysis
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="space-y-6">
            <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 shadow-lg border border-blue-100">
              <label className="block text-base font-semibold text-gray-800 mb-4">
                Upload {isBatchMode ? 'Multiple CT Scan Images' : 'CT Scan Image'}
              </label>
              <input
                type="file"
                onChange={(e) => isBatchMode ? setSelectedFiles(e.target.files) : setSelectedFile(e.target.files?.[0] || null)}
                accept="image/*"
                multiple={isBatchMode}
                className="block w-full text-sm text-gray-600 file:mr-4 file:py-3 file:px-6 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-gradient-to-r file:from-blue-600 file:to-indigo-600 file:text-white hover:file:from-blue-700 hover:file:to-indigo-700"
              />
              <button
                onClick={isBatchMode ? handleBatchPredict : handleSinglePredict}
                disabled={isBatchMode ? !selectedFiles?.length : !selectedFile || loading}
                className="w-full mt-6 px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all duration-300 shadow-md"
              >
                {loading ? 'Processing...' : `Analyze ${isBatchMode ? 'Images' : 'Image'}`}
              </button>
            </div>

            {error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-xl text-red-600">
                {error}
              </div>
            )}

            {/* Batch Results Display */}
            {isBatchMode && batchPredictions.length > 0 && (
              <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 shadow-lg border border-blue-100">
                <h3 className="text-xl font-bold text-gray-800 mb-6">Batch Analysis Results</h3>
                <div className="space-y-4">
                  {batchPredictions.map((pred, index) => (
                    <div key={index} className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-100">
                      <h4 className="font-semibold text-gray-800 mb-3">Image {index + 1}</h4>
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-gray-600">Predicted Class</span>
                          <span className="font-semibold text-blue-600">
                            {formatClassName(pred.prediction.predicted_class)}
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-gray-600">Confidence</span>
                          <span className="font-semibold text-indigo-600">
                            {(pred.prediction.confidence * 100).toFixed(2)}%
                          </span>
                        </div>
                        {pred.prediction.probabilities && (
                          <div className="mt-2">
                            <div className="space-y-2">
                              {Object.entries(pred.prediction.probabilities)
                                .sort(([, a], [, b]) => b - a)
                                .map(([key, value]) => (
                                  <div key={key} className="flex items-center space-x-2">
                                    <div className="w-full bg-gray-200 rounded-full h-2">
                                      <div 
                                        className="bg-gradient-to-r from-blue-600 to-indigo-600 h-2 rounded-full" 
                                        style={{ width: `${value * 100}%` }}
                                      />
                                    </div>
                                    <span className="text-sm text-gray-600 min-w-[60px] text-right">
                                      {(value * 100).toFixed(1)}%
                                    </span>
                                    <span className="text-sm font-medium text-gray-800">
                                      {formatClassName(key)}
                                    </span>
                                  </div>
                                ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Single Image Results - existing code remains the same */}
            {!isBatchMode && prediction && prediction.prediction && (
              <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 shadow-lg border border-blue-100">
                <h3 className="text-xl font-bold text-gray-800 mb-6">Analysis Results</h3>
                <div className="space-y-4">
                  <div className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border border-blue-100">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-600">Predicted Class</span>
                      <span className="font-semibold text-blue-600 text-lg">
                        {formatClassName(prediction.prediction.predicted_class)}
                      </span>
                    </div>
                    <div className="mt-2 flex justify-between items-center">
                      <span className="text-gray-600">Confidence Level</span>
                      <span className="font-semibold text-indigo-600 text-lg">
                        {(prediction.prediction.confidence * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>

                  {prediction.prediction.probabilities && (
                    <div className="mt-6">
                      <h4 className="font-semibold text-gray-800 mb-3">Probability Distribution</h4>
                      <div className="space-y-2">
                        {Object.entries(prediction.prediction.probabilities)
                          .sort(([, a], [, b]) => b - a)
                          .map(([key, value]) => (
                            <div key={key} className="flex items-center space-x-2">
                              <div className="w-full bg-gray-200 rounded-full h-2.5">
                                <div 
                                  className="bg-gradient-to-r from-blue-600 to-indigo-600 h-2.5 rounded-full" 
                                  style={{ width: `${value * 100}%` }}
                                />
                              </div>
                              <span className="text-sm text-gray-600 min-w-[100px] text-right">
                                {(value * 100).toFixed(1)}%
                              </span>
                              <span className="text-sm font-medium text-gray-800 min-w-[120px]">
                                {formatClassName(key)}
                              </span>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Right Panel - Enhanced Visualizations */}
          <div className="space-y-6">
            {prediction && prediction.plots && (
              <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 shadow-lg border border-blue-100">
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-xl font-bold text-gray-800">Visualization Analysis</h3>
                  <div className="flex gap-2">
                    {Object.keys(prediction.plots).map((key, index) => (
                      <button
                        key={key}
                        className={`px-3 py-1 text-sm rounded-full transition-all duration-300 ${
                          index === 0 
                            ? 'bg-gradient-to-r from-blue-600 to-indigo-600 text-white' 
                            : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                        }`}
                      >
                        {formatClassName(key)}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(prediction.plots).map(([key, base64Data]) => (
                    <div 
                      key={key} 
                      className="group relative overflow-hidden rounded-lg border border-blue-100 transition-all duration-300 hover:shadow-xl"
                    >
                      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-black/50 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                      <img 
                        src={`data:image/png;base64,${base64Data}`}
                        alt={`${formatClassName(key)} Visualization`}
                        className="w-full h-auto object-cover transform group-hover:scale-105 transition-transform duration-300"
                      />
                      <div className="absolute bottom-0 left-0 right-0 p-4 text-white transform translate-y-full group-hover:translate-y-0 transition-transform duration-300">
                        <h4 className="text-sm font-semibold">
                          {formatClassName(key)}
                        </h4>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="mt-4 p-4 bg-blue-50/50 rounded-lg">
                  <p className="text-sm text-gray-600">
                    Hover over images to see detailed visualization information. 
                    Each image represents different aspects of the analysis.
                  </p>
                </div>
              </div>
            )}
                 </div>
        </div>
      </div>
    </main>
  );
}
