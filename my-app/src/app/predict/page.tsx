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

export default function PredictPage() {
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

  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 text-white p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-center justify-between mb-8">
          <h1 className="text-3xl font-bold">Single Image Analysis</h1>
          <Link 
            href="/" 
            className="px-4 py-2 bg-gray-700 rounded-lg hover:bg-gray-600 transition-colors"
          >
            Back to Home
          </Link>
        </div>

        <div className="bg-gray-800 rounded-xl p-6 shadow-lg border border-gray-700">
          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium mb-2">
                Upload CT Scan Image
              </label>
              <input
                type="file"
                onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
                accept="image/*"
                className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-500 file:text-white hover:file:bg-blue-600"
              />
            </div>

            <button
              onClick={handleSinglePredict}
              disabled={!selectedFile || loading}
              className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
            >
              {loading ? 'Processing...' : 'Analyze Image'}
            </button>

            {error && (
              <div className="p-4 bg-red-900/50 border border-red-700 rounded-lg text-red-200">
                {error}
              </div>
            )}

            {prediction && (
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