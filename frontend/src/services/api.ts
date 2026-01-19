/**
 * API Service
 * HTTP client for backend API calls
 */

import axios from 'axios';
import { Prediction, BacktestResult, ModelInfo, BacktestRequest, StockData } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add any auth tokens here if needed
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Server responded with error
      console.error('API Error:', error.response.data);
    } else if (error.request) {
      // Request made but no response
      console.error('Network Error:', error.message);
    }
    return Promise.reject(error);
  }
);

export const predictionAPI = {
  /**
   * Get prediction for a single ticker
   */
  getPrediction: async (
    ticker: string, 
    modelName?: string
  ): Promise<Prediction> => {
    const response = await api.post('/api/predictions/predict', {
      ticker,
      model_name: modelName || 'Random_Forest_Classifier',
      target_type: 'direction',
      return_confidence: true,
    });
    return response.data;
  },

  /**
   * Get predictions for multiple tickers
   */
  getBatchPredictions: async (tickers: string[]): Promise<Prediction[]> => {
    const response = await api.post('/api/predictions/predict/batch', {
      tickers,
    });
    return response.data.predictions;
  },

  /**
   * Get available models for a ticker
   */
  getAvailableModels: async (ticker: string): Promise<ModelInfo[]> => {
    const response = await api.get(`/api/predictions/models/${ticker}`);
    return response.data.models;
  },
};

export const backtestAPI = {
  /**
   * Run backtest for tickers
   */
  runBacktest: async (request: BacktestRequest): Promise<BacktestResult> => {
    const response = await api.post('/api/backtest/run', request);
    // If multiple tickers, return first result or handle differently
    if (request.tickers.length === 1) {
      return response.data.results[request.tickers[0]];
    }
    return response.data;
  },

  /**
   * Get cached backtest results
   */
  getBacktestResults: async (ticker: string): Promise<BacktestResult> => {
    const response = await api.get(`/api/backtest/results/${ticker}`);
    return response.data;
  },
};

export const dataAPI = {
  /**
   * Get stock data for a ticker
   * Returns the data array directly
   */
  getStockData: async (
    ticker: string,
    period: string = '1y',
    interval: string = '1d'
  ): Promise<StockData[]> => {
    const response = await api.get(`/api/data/ticker/${ticker}`, {
      params: { period, interval },
    });
    // Extract the nested data array from the response
    return response.data.data || [];
  },

  /**
   * Get stock data for multiple tickers
   */
  getBatchStockData: async (
    tickers: string[],
    period: string = '1y',
    interval: string = '1d'
  ) => {
    const response = await api.post('/api/data/batch', {
      tickers,
      period,
      interval,
    });
    return response.data;
  },
};

export default api;
