import asyncio
print("Starting pi_server.py...")
import websockets
import cv2
import numpy as np
import json
import argparse
from test_model import YOLOInference

# Initialize model globally (will be loaded in main)
model = None

import easyocr

# Initialize OCR reader globally (lazy load or in main)
reader = None

async def handle_connection(websocket):
    print("Client connected")
    try:
        async for message in websocket:
            try:
                # Decode image in thread pool to avoid blocking
                loop = asyncio.get_running_loop()
                image = await loop.run_in_executor(None, decode_image, message)
                
                if image is None:
                    print("Failed to decode image")
                    continue

                # Run inference in thread pool
                boxes, scores, class_ids = await loop.run_in_executor(None, model.run_inference, image)
                
                # Format results and run OCR
                results = []
                for box, score, class_id in zip(boxes, scores, class_ids):
                    # Crop the license plate
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Clamp coordinates
                    h, w, _ = image.shape
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    text_content = ""
                    if x2 > x1 and y2 > y1:
                        plate_crop = image[y1:y2, x1:x2]
                        # Run OCR on crop
                        # detail=0 returns just the list of text strings
                        ocr_result = await loop.run_in_executor(None, reader.readtext, plate_crop, 0) # detail=0 -> simply returns all texts
                        if ocr_result:
                             # Join standard output (it might be a list of strings)
                             # easyocr with detail=0 returns a list of strings
                             text_content = " ".join(ocr_result)
                    
                    results.append({
                        "class_id": int(class_id),
                        "score": float(score),
                        "box": box.tolist(),
                        "text": text_content
                    })
                
                # Send back JSON
                await websocket.send(json.dumps(results))
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

def decode_image(message):
    np_arr = np.frombuffer(message, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

async def main(model_path, port):
    global model, reader
    print(f"Loading model from {model_path}...")
    model = YOLOInference(model_path)
    print("Model loaded.")
    
    print("Loading OCR reader...")
    reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if running on proper GPU machine
    print("OCR reader loaded.")
    
    print(f"Starting server on port {port}...")
    async with websockets.serve(handle_connection, "0.0.0.0", port):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fixed_model.onnx", help="Path to ONNX model")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    args = parser.parse_args()
    
    asyncio.run(main(args.model, args.port))
