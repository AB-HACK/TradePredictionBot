# Trade Prediction Bot - Backend API

FastAPI backend with WebSocket support for the Trade Prediction Bot.

## Setup

### Install Dependencies

First, install the main project dependencies (from project root):

```bash
pip install -r requirements.txt
```

Then install backend-specific dependencies:

```bash
cd backend
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the backend directory (optional):

```env
PORT=8000
HOST=0.0.0.0
```

### Run the Server

From the `backend` directory:

```bash
python -m app.main
```

Or using uvicorn directly:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- WebSocket: ws://localhost:8000/ws

## API Endpoints

### Predictions

- `POST /api/predictions/predict` - Get prediction for a ticker
- `POST /api/predictions/predict/batch` - Get predictions for multiple tickers
- `GET /api/predictions/models/{ticker}` - Get available models for a ticker

### Data

- `GET /api/data/ticker/{ticker}` - Get stock data for a ticker
- `POST /api/data/batch` - Get stock data for multiple tickers

### Backtest

- `POST /api/backtest/run` - Run backtest for tickers
- `GET /api/backtest/results/{ticker}` - Get cached backtest results

### Models

- `GET /api/models/list` - List available models
- `GET /api/models/info/{ticker}/{model_name}` - Get model information

## WebSocket

Connect to `ws://localhost:8000/ws` for real-time updates.

Message format:

```json
{
  "type": "subscribe_predictions",
  "ticker": "AAPL"
}
```

The server will send prediction updates every 30 seconds for subscribed tickers.

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── api/
│   │   ├── routes/          # API route handlers
│   │   └── websocket.py     # WebSocket manager
│   └── services/            # Business logic
└── requirements.txt         # Backend dependencies
```
