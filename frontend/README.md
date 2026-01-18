# Trade Prediction Bot - Frontend

React + TypeScript + Tailwind CSS frontend for the Trade Prediction Bot.

## Setup

### Install Dependencies

```bash
npm install
```

### Start Development Server

```bash
npm start
```

Runs on [http://localhost:3000](http://localhost:3000)

### Build for Production

```bash
npm run build
```

## Environment Variables

Create a `.env` file in the frontend directory:

```
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000/ws
```

## Project Structure

```
src/
├── components/       # React components
│   ├── Dashboard/   # Main dashboard
│   ├── Chart/       # Stock charts
│   └── PredictionCard/ # Prediction display
├── services/        # API and WebSocket services
├── hooks/          # Custom React hooks
├── types/          # TypeScript type definitions
└── App.tsx         # Main app component
```
