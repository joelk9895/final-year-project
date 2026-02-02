import asyncio
import websockets
import cv2
import json
import argparse

async def send_image(uri, image_path):
    async with websockets.connect(uri) as websocket:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            return
            
        # Encode as JPEG
        _, buffer = cv2.imencode('.jpg', img)
        byte_data = buffer.tobytes()
        
        print(f"Sending image ({len(byte_data)} bytes)...")
        await websocket.send(byte_data)
        
        response = await websocket.recv()
        print(f"Received response: {response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", type=str, default="ws://localhost:8765")
    parser.add_argument("--image", type=str, required=True)
    args = parser.parse_args()
    
    asyncio.run(send_image(args.uri, args.image))
