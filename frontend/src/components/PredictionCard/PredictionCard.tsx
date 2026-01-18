/**
 * Prediction Card Component
 * Displays prediction results in a card format
 */

import React from 'react';
import { Prediction } from '../../types';

interface PredictionCardProps {
  prediction: Prediction;
  loading?: boolean;
}

export const PredictionCard: React.FC<PredictionCardProps> = ({ 
  prediction, 
  loading = false 
}) => {
  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6 border-l-4 border-gray-400 animate-pulse">
        <div className="h-6 bg-gray-200 rounded w-1/4 mb-4"></div>
        <div className="h-4 bg-gray-200 rounded w-1/2 mb-2"></div>
        <div className="h-4 bg-gray-200 rounded w-1/3"></div>
      </div>
    );
  }

  if (!prediction) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6 border-l-4 border-gray-400">
        <p className="text-gray-600">No prediction available</p>
      </div>
    );
  }

  const isUp = prediction.direction === 'UP';
  const confidenceColor = prediction.confidence
    ? prediction.confidence > 0.7
      ? 'text-green-600'
      : prediction.confidence > 0.5
      ? 'text-yellow-600'
      : 'text-red-600'
    : 'text-gray-600';

  return (
    <div className={`bg-white rounded-lg shadow-md p-6 border-l-4 ${
      isUp ? 'border-green-500' : 'border-red-500'
    }`}>
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-2xl font-bold text-gray-800">{prediction.ticker}</h3>
        <span
          className={`px-3 py-1 rounded-full text-sm font-semibold ${
            isUp
              ? 'bg-green-100 text-green-800'
              : 'bg-red-100 text-red-800'
          }`}
        >
          {prediction.direction}
        </span>
      </div>

      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-gray-600">Prediction:</span>
          <span className={`text-lg font-bold ${
            isUp ? 'text-green-600' : 'text-red-600'
          }`}>
            {prediction.prediction.toFixed(4)}
          </span>
        </div>

        {prediction.confidence !== null && (
          <div className="flex justify-between items-center">
            <span className="text-gray-600">Confidence:</span>
            <span className={`font-semibold ${confidenceColor}`}>
              {(prediction.confidence * 100).toFixed(1)}%
            </span>
          </div>
        )}

        <div className="pt-3 border-t border-gray-200">
          <div className="flex justify-between text-sm text-gray-500">
            <span>Updated:</span>
            <span>{new Date(prediction.timestamp).toLocaleString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
};
