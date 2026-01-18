/**
 * TypeScript type definitions for Trade Prediction Bot
 */

export interface Prediction {
  ticker: string;
  prediction: number;
  confidence: number | null;
  direction: "UP" | "DOWN";
  timestamp: string;
}

export interface BacktestResult {
  total_return_pct: number;
  annualized_return_pct: number;
  volatility_pct: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown_pct: number;
  calmar_ratio: number;
  win_rate_pct: number;
  total_trades: number;
  avg_win: number;
  avg_loss: number;
  profit_factor: number;
  portfolio_history: Array<{
    date: string;
    portfolio_value: number;
    returns: number;
  }>;
}

export interface ModelInfo {
  name: string;
  accuracy?: number;
  rmse?: number;
  r2?: number;
}

export interface StockData {
  timestamp: string;
  Open: number;
  High: number;
  Low: number;
  Close: number;
  Volume: number;
}

export interface BacktestRequest {
  tickers: string[];
  target_type?: string;
  signal_type?: string;
  initial_capital?: number;
}
