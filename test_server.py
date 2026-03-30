"""
Minimal WebSocket test server — no ML dependencies.
Run: python test_server.py
Then connect the app to ws://<mac-ip>:8765
"""
import asyncio
import websockets
import socket

def get_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "unknown"
    finally:
        s.close()

async def handler(websocket):
    print(f"✅ iPhone connected from {websocket.remote_address}")
    frame_count = 0
    try:
        async for message in websocket:
            frame_count += 1
            if isinstance(message, bytes):
                print(f"  Frame {frame_count}: {len(message)} bytes received")
            else:
                print(f"  Text: {message}")
            await websocket.send(f'{{"type":"ok","frame":{frame_count}}}')
    except websockets.exceptions.ConnectionClosed:
        print(f"  Disconnected after {frame_count} frames")

async def main():
    ip = get_ip()
    print(f"\n{'='*50}")
    print(f"  Test server ready")
    print(f"  Connect app to:  ws://{ip}:8765")
    print(f"{'='*50}\n")
    async with websockets.serve(handler, "0.0.0.0", 8765):
        await asyncio.Future()

asyncio.run(main())
