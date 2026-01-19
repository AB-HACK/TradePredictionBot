/**
 * Custom hook for predictions
 */

import { useState, useEffect, useCallback } from 'react';
import { Prediction } from '../types';
import { predictionAPI } from '../services/api';
import { websocketService } from '../services/websocket';

interface UsePredictionsOptions {
  ticker: string;
  modelName?: string;
  useWebSocket?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export const usePredictions = (options: UsePredictionsOptions) => {
  const {
    ticker,
    modelName,
    useWebSocket = false,
    autoRefresh = false,
    refreshInterval = 30000,
  } = options;

  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchPrediction = useCallback(async () => {
    if (!ticker) return;

    setLoading(true);
    setError(null);

    try {
      const result = await predictionAPI.getPrediction(ticker, modelName);
      setPrediction(result);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch prediction');
      console.error('Prediction fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, [ticker, modelName]);

  // Initial fetch
  useEffect(() => {
    fetchPrediction();
  }, [fetchPrediction]);

  // WebSocket subscription
  useEffect(() => {
    if (useWebSocket && ticker) {
      websocketService.connect(ticker, (data: Prediction) => {
        setPrediction(data);
      });

      return () => {
        websocketService.disconnect();
      };
    }
  }, [ticker, useWebSocket]);

  // Auto refresh
  useEffect(() => {
    if (autoRefresh && !useWebSocket) {
      const interval = setInterval(() => {
        fetchPrediction();
      }, refreshInterval);

      return () => clearInterval(interval);
    }
  }, [autoRefresh, useWebSocket, refreshInterval, fetchPrediction]);

  return {
    prediction,
    loading,
    error,
    refetch: fetchPrediction,
  };
};
