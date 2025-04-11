'use client';

import { useState } from 'react';
import Link from 'next/link';

interface PredictionResponse {
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
  plot_data?: string;
  visualization_data?: string; // Additional visualization data
}

interface BatchPredictionResponse {
  predictions: PredictionResponse[];
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
      return className; // Return original string if formatting fails
    }
  };

  const handleSinglePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      // Make both requests concurrently
      const [predictionResponse, visualizationResponse] = await Promise.all([
        fetch('http://localhost:8000/predict', {
          method: 'POST',
          body: formData,
          credentials: 'include',
          headers: { 'Accept': 'application/json' },
          mode: 'cors'
        }),
        fetch('http://localhost:8000/visualize', {
          method: 'POST',
          body: formData,
          credentials: 'include',
          headers: { 'Accept': 'application/json' },
          mode: 'cors'
        })
      ]);

      if (!predictionResponse.ok || !visualizationResponse.ok) {
        throw new Error(`Request failed: ${predictionResponse.status} ${visualizationResponse.status}`);
      }

      const [predictionData, visualizationData] = await Promise.all([
        predictionResponse.json(),
        visualizationResponse.json()
      ]);

      // Combine both results
      setPrediction({
        ...predictionData,
        visualization_data: visualizationData.visualization
      });
    } catch (error) {
      setError('Error processing image. Please try again.');
      console.error('Error:', error);
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
      setBatchPredictions(data.predictions);
    } catch (error) {
      setError('Error processing batch. Please try again.');
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">Image Analysis</h1>
          <Link href="/" className="px-4 py-2 bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors">
            Back to Home
          </Link>
        </div>

        <div className="bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-700">
          <div className="space-y-6">
            {/* Mode Switch */}
            <div className="flex gap-4">
              <button
                onClick={() => {
                  setIsBatchMode(false);
                  setSelectedFiles(null);
                  setBatchPredictions([]);
                }}
                className={`px-4 py-2 rounded-lg transition-colors ${
                  !isBatchMode ? 'bg-blue-500 text-white' : 'bg-gray-700 text-gray-300'
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
                className={`px-4 py-2 rounded-lg transition-colors ${
                  isBatchMode ? 'bg-blue-500 text-white' : 'bg-gray-700 text-gray-300'
                }`}
              >
                Batch Processing
              </button>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">
                Upload {isBatchMode ? 'CT Scan Images' : 'CT Scan Image'}
              </label>
              <input
                type="file"
                onChange={(e) => isBatchMode ? setSelectedFiles(e.target.files) : setSelectedFile(e.target.files?.[0] || null)}
                accept="image/*"
                multiple={isBatchMode}
                className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-500 file:text-white hover:file:bg-blue-600"
              />
            </div>

            <button
              onClick={isBatchMode ? handleBatchPredict : handleSinglePredict}
              disabled={isBatchMode ? !selectedFiles?.length : !selectedFile || loading}
              className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Processing...' : `Analyze ${isBatchMode ? 'Images' : 'Image'}`}
            </button>

            {/* Error Display */}
            {error && (
              <div className="p-4 bg-red-900/50 border border-red-700 rounded-lg text-red-200">
                {error}
              </div>
            )}

            {/* Batch Results Display */}
            {isBatchMode && batchPredictions.length > 0 && (
              <div className="mt-6 space-y-6">
                <h3 className="text-xl font-semibold mb-4">Batch Results</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {batchPredictions.map((pred, index) => (
                    <div key={index} className="p-4 bg-gray-700 rounded-lg">
                      <h4 className="font-medium text-lg mb-3">Image {index + 1}</h4>
                      <div className="space-y-2">
                        <div className="flex justify-between items-center p-2 bg-gray-800 rounded">
                          <span>Class:</span>
                          <span className="font-semibold text-blue-400">
                            {formatClassName(pred.predicted_class)}
                          </span>
                        </div>
                        <div className="flex justify-between items-center p-2 bg-gray-800 rounded">
                          <span>Confidence:</span>
                          <span className="font-semibold text-green-400">
                            {(pred.confidence * 100).toFixed(2)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Single Image Results - existing code remains the same */}
            {!isBatchMode && prediction && (
              <div className="mt-6 space-y-6">
                <div className="p-6 bg-gray-700 rounded-lg">
                  <h3 className="text-xl font-semibold mb-4">Results</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center p-3 bg-gray-800 rounded">
                      <span>Predicted Class:</span>
                      <span className="font-semibold text-blue-400">
                        {formatClassName(prediction.predicted_class)}
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center p-3 bg-gray-800 rounded">
                      <span>Confidence:</span>
                      <span className="font-semibold text-green-400">
                        {(prediction.confidence * 100).toFixed(2)}%
                      </span>
                    </div>

                    {prediction.probabilities && (
                      <div className="mt-4">
                        <h4 className="font-semibold mb-2">Class Probabilities:</h4>
                        <div className="space-y-2">
                          {Object.entries(prediction.probabilities)
                            .sort(([, a], [, b]) => b - a)
                            .map(([key, value]) => (
                              <div key={key} className="flex justify-between items-center p-2 bg-gray-800 rounded">
                                <span>{formatClassName(key)}</span>
                                <span>{(value * 100).toFixed(2)}%</span>
                              </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>

                {prediction.plot_data && (
                  <div className="mt-6 p-6 bg-gray-700 rounded-lg">
                    <h3 className="text-xl font-semibold mb-4">Visualization</h3>
                    <div className="w-full overflow-hidden rounded-lg">
                      <img 
                        src={`data:image/png;base64,${prediction.plot_data}`}
                        alt="Prediction Visualization"
                        className="w-full h-auto"
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}