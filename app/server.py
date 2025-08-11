import asyncio
import websockets
from config.settings import config, WEBSOCKET_PORT
from handlers.websocket_handler import WebSocketHandler
from core.utils import printer

async def main():
    print_startup_info()
    
    async def handler(websocket, path=None):
        printer(f"[CONNECTION] New client from {websocket.remote_address}", "info")
        ws_handler = WebSocketHandler(websocket)
        await ws_handler.handle_connection()

    async with websockets.serve(handler, "0.0.0.0", WEBSOCKET_PORT):
        printer(f"[INFO] Server started on ws://0.0.0.0:{WEBSOCKET_PORT}", "info")
        await asyncio.Future()

def print_startup_info():
    print("*" * 60)
    print(f"[INFO] Bedrock Model: {config['model_id']}")
    print(f"[INFO] Polly Sample Rate: {config['polly']['SampleRate']}Hz")
    print("[INFO] Voice Assistant Server is ready.")
    print("*" * 60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nApplication terminated by user.")
