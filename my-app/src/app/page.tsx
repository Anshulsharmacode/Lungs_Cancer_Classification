'use client';

import { useState } from 'react';
import Link from 'next/link';

interface PredictionResponse {
  predicted_class: string;
  confidence: number;
  probabilities: Record<string, number>;
}

export default function PredictPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
      });
      
      if (!response.ok) {
        throw new Error('Prediction failed');
      }
      
      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      setError('Error processing image. Please try again.');
      console.error('Error predicting:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatClassName = (className: string) => {
    return className
      .split('_')
      .map(word => word.split('.').join(' '))
      .join(' - ')
      .replace(/(^|\s)\S/g, letter => letter.toUpperCase());
  };

  return (
    <main className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto p-8">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold text-gray-800">Single Image Analysis</h1>
          <Link 
            href="/" 
            className="px-4 py-2 bg-white rounded-lg hover:bg-gray-100 transition-colors border border-gray-200 text-gray-600"
          >
            Back to Home
          </Link>
        </div>

        <div className="bg-white rounded-xl p-6 shadow-lg border border-gray-200">
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload CT Scan Image
              </label>
              <input
                type="file"
                onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                accept="image/*"
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-500 file:text-white hover:file:bg-blue-600"
              />
            </div>

            <button
              onClick={handleSinglePredict}
              disabled={!selectedFile || loading}
              className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Processing...' : 'Analyze Image'}
            </button>

            {error && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-600">
                {error}
              </div>
            )}

            {prediction && (
              <div className="mt-6 space-y-6">
                <div className="p-6 bg-blue-50 rounded-lg border border-blue-100">
                  <h3 className="text-xl font-semibold mb-4 text-gray-800">Primary Prediction</h3>
                  <div className="space-y-3">
                    <div className="flex justify-between items-center p-3 bg-white rounded-lg border border-blue-100">
                      <span className="text-gray-600">Predicted Class:</span>
                      <span className="font-semibold text-blue-600">
                        {formatClassName(prediction.predicted_class)}
                      </span>
                    </div>
                    <div className="flex justify-between items-center p-3 bg-white rounded-lg border border-blue-100">
                      <span className="text-gray-600">Confidence:</span>
                      <span className="font-semibold text-blue-600">
                        {(prediction.confidence * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>
                </div>

                <div className="p-6 bg-gray-50 rounded-lg border border-gray-200">
                  <h3 className="text-xl font-semibold mb-4 text-gray-800">All Probabilities</h3>
                  <div className="space-y-2">
                    {Object.entries(prediction.probabilities)
                      .sort(([, a], [, b]) => b - a) // Sort by probability in descending order
                      .map(([key, value]) => (
                        <div key={key} 
                          className="flex justify-between items-center p-3 bg-white rounded-lg border border-gray-200 hover:border-blue-200 transition-colors">
                          <span className="text-gray-600">{formatClassName(key)}:</span>
                          <div className="flex items-center">
                            <div className="w-32 h-2 bg-gray-200 rounded-full mr-3">
                              <div 
                                className="h-2 bg-blue-500 rounded-full"
                                style={{ width: `${value * 100}%` }}
                              />
                            </div>
                            <span className="font-semibold text-gray-700">
                              {(value * 100).toFixed(2)}%
                            </span>
                          </div>
                        </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}