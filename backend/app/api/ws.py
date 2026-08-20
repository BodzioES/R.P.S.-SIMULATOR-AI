from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..sim.state import manager

ws_router = APIRouter()


@ws_router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    manager.clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        manager.clients.discard(websocket)