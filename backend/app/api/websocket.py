"""
WebSocket Manager
Handles WebSocket connections for real-time updates
"""

from fastapi import WebSocket
from typing import List, Dict
import json
import asyncio
import logging

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: Dict[WebSocket, List[str]] = {}  # tickers per connection
        self.prediction_service = None
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = []
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific WebSocket connection"""
        try:
            await websocket.send_text(message)
        except (ConnectionError, RuntimeError) as e:
            logger.warning(f"Connection error sending message: {e}")
            self.disconnect(websocket)
        except Exception as e:
            logger.error(f"Unexpected error sending message: {e}", exc_info=True)
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        message_str = json.dumps(message)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except (ConnectionError, RuntimeError) as e:
                logger.warning(f"Connection error broadcasting: {e}")
                disconnected.append(connection)
            except Exception as e:
                logger.error(f"Unexpected error broadcasting: {e}", exc_info=True)
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)
    
    async def handle_message(self, websocket: WebSocket, message: str):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            if message_type == "subscribe_predictions":
                ticker = data.get("ticker")
                if ticker:
                    if websocket not in self.subscriptions:
                        self.subscriptions[websocket] = []
                    if ticker not in self.subscriptions[websocket]:
                        self.subscriptions[websocket].append(ticker)
                    await self._start_prediction_stream(websocket, ticker)
            
            elif message_type == "unsubscribe_predictions":
                ticker = data.get("ticker")
                if websocket in self.subscriptions and ticker in self.subscriptions[websocket]:
                    self.subscriptions[websocket].remove(ticker)
            
            elif message_type == "get_prediction":
                ticker = data.get("ticker")
                if ticker and self.prediction_service:
                    result = await self.prediction_service.get_prediction(ticker)
                    await self.send_personal_message(json.dumps(result), websocket)
        
        except json.JSONDecodeError as e:
            await self.send_personal_message(
                json.dumps({"error": "Invalid JSON format"}),
                websocket
            )
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}", exc_info=True)
            await self.send_personal_message(
                json.dumps({"error": str(e)}),
                websocket
            )
    
    async def _start_prediction_stream(self, websocket: WebSocket, ticker: str):
        """Start streaming predictions for a ticker"""
        if not self.prediction_service:
            # Lazy import to avoid circular dependencies
            from app.services.prediction_service import PredictionService
            self.prediction_service = PredictionService()
        
        # Start background task for streaming
        asyncio.create_task(self._stream_predictions(websocket, ticker))
    
    async def _stream_predictions(self, websocket: WebSocket, ticker: str):
        """Stream predictions every 30 seconds"""
        while websocket in self.active_connections:
            try:
                if websocket in self.subscriptions and ticker in self.subscriptions[websocket]:
                    result = await self.prediction_service.get_prediction(ticker)
                    await self.send_personal_message(json.dumps(result), websocket)
                    await asyncio.sleep(30)  # Update every 30 seconds
                else:
                    break  # Unsubscribed
            except (ValueError, KeyError) as e:
                logger.warning(f"Data error streaming predictions for {ticker}: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error streaming predictions for {ticker}: {e}", exc_info=True)
                break

# Global WebSocket manager instance
websocket_manager = WebSocketManager()
