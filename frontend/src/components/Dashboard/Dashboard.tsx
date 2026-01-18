/**
 * Main Dashboard Component
 * Main page showing predictions, charts, and controls
 */

import React, { useState, useEffect } from 'react';
import { PredictionCard } from '../PredictionCard/PredictionCard';
import { StockChart } from '../Chart/StockChart';
import { usePredictions } from '../../hooks/usePredictions';
import { dataAPI } from '../../services/api';
import { StockData } from '../../types';

export const Dashboard: React.FC = () => {
  const [ticker, setTicker] = useState('AAPL');
  const [stockData, setStockData] = useState<StockData[]>([]);
  const [loadingData, setLoadingData] = useState(false);
  
  const {
    prediction,
    loading: loadingPrediction,
    error: predictionError,
    refetch: refetchPrediction,
  } = usePredictions({
    ticker,
    useWebSocket: true,
    autoRefresh: true,
  });

  useEffect(() => {
    loadStockData();
  }, [ticker]);

  const loadStockData = async () => {
    setLoadingData(true);
    try {
      const stockDataArray = await dataAPI.getStockData(ticker, '1y', '1d');
      if (stockDataArray && stockDataArray.length > 0) {
        setStockData(stockDataArray);
      }
    } catch (error) {
      console.error('Error loading stock data:', error);
    } finally {
      setLoadingData(false);
    }
  };

  const handleTickerChange = (newTicker: string) => {
    const upperTicker = newTicker.toUpperCase().trim();
    if (upperTicker.length > 0) {
      setTicker(upperTicker);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4 md:p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Trading Dashboard
          </h1>
          <p className="text-gray-600">
            Real-time stock predictions and analysis
          </p>
        </div>

        {/* Ticker Input */}
        <div className="mb-6 bg-white rounded-lg shadow-md p-4">
          <div className="flex flex-col sm:flex-row gap-4 items-center">
            <label htmlFor="ticker" className="text-gray-700 font-medium">
              Stock Ticker:
            </label>
            <input
              id="ticker"
              type="text"
              value={ticker}
              onChange={(e) => handleTickerChange(e.target.value)}
              onKeyPress={(e) => {
                if (e.key === 'Enter') {
                  handleTickerChange(ticker);
                }
              }}
              placeholder="Enter ticker symbol (e.g., AAPL)"
              className="px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 flex-1 max-w-xs"
            />
            <button
              onClick={() => {
                refetchPrediction();
                loadStockData();
              }}
              className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
            >
              Refresh
            </button>
          </div>
        </div>

        {/* Error Display */}
        {predictionError && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-red-800">{predictionError}</p>
          </div>
        )}

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Prediction Card */}
          <div className="lg:col-span-1">
            <PredictionCard
              prediction={prediction}
              loading={loadingPrediction}
            />
          </div>

          {/* Chart */}
          <div className="lg:col-span-2">
            {loadingData ? (
              <div className="bg-white rounded-lg shadow-md p-6">
                <div className="animate-pulse">
                  <div className="h-6 bg-gray-200 rounded w-1/4 mb-4"></div>
                  <div className="h-96 bg-gray-200 rounded"></div>
                </div>
              </div>
            ) : (
              <StockChart ticker={ticker} data={stockData} showVolume={true} />
            )}
          </div>
        </div>

        {/* Additional Info */}
        <div className="bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold text-gray-800 mb-4">
            About {ticker}
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-gray-600">Data Points</p>
              <p className="text-lg font-semibold">{stockData.length}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Latest Price</p>
              <p className="text-lg font-semibold">
                ${stockData[stockData.length - 1]?.Close.toFixed(2) || 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Prediction Status</p>
              <p className="text-lg font-semibold">
                {loadingPrediction ? 'Loading...' : prediction?.direction || 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Confidence</p>
              <p className="text-lg font-semibold">
                {prediction?.confidence
                  ? `${(prediction.confidence * 100).toFixed(1)}%`
                  : 'N/A'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
