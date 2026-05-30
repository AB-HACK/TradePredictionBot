/**
 * Stock Chart Component
 * Displays historical price and volume data using recharts
 */

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';
import { StockData } from '../../types';

interface StockChartProps {
  ticker: string;
  data: StockData[];
  showVolume?: boolean;
}

export const StockChart: React.FC<StockChartProps> = ({
  ticker,
  data,
  showVolume = false,
}) => {
  if (!data || data.length === 0) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4">{ticker} Price Chart</h3>
        <p className="text-gray-600">No chart data available</p>
      </div>
    );
  }

  const chartData = data.map((point) => ({
    date: new Date(point.timestamp).toLocaleDateString(),
    close: point.Close,
    volume: point.Volume,
  }));

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h3 className="text-xl font-bold text-gray-800 mb-4">{ticker} Price Chart</h3>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fontSize: 12 }} />
          <YAxis domain={['auto', 'auto']} tick={{ fontSize: 12 }} />
          <Tooltip />
          <Line type="monotone" dataKey="close" stroke="#2563eb" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>

      {showVolume && (
        <div className="mt-6">
          <h4 className="text-lg font-semibold text-gray-800 mb-2">Volume</h4>
          <ResponsiveContainer width="100%" height={150}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip />
              <Bar dataKey="volume" fill="#93c5fd" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};
