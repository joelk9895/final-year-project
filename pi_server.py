import asyncio
import time
print("Starting pi_server.py...")
import websockets
import cv2
import numpy as np
import json
import argparse
import sqlite3
import os
import socket
from aiohttp import web
from test_model import YOLOInference

model = None

import easyocr

reader = None

import re

# ─── Connected clients & latest frame ─────────────────
connected_clients = set()
_latest_frame_jpg = None  # JPEG bytes of latest processed frame


DB_PATH = "vehicles.db"

# ─── OCR cache / cooldown ─────────────────────────────
OCR_COOLDOWN_SECONDS = 3.0          # skip OCR if we got a valid read this recently
_last_ocr_plate = None               # cached plate text
_last_ocr_time  = 0.0                # timestamp of last successful OCR

# ─── Persistent DB connection ─────────────────────────
_db_conn = None

def _get_db():
    """Return the persistent SQLite connection (created once)."""
    global _db_conn
    if _db_conn is None:
        _db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _db_conn.row_factory = sqlite3.Row
    return _db_conn

def init_database():
    """Create the vehicles table and seed initial data if the DB doesn't exist."""
    is_new = not os.path.exists(DB_PATH)
    conn = _get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            plate TEXT PRIMARY KEY,
            owner TEXT NOT NULL,
            phone TEXT DEFAULT '',
            balance REAL NOT NULL DEFAULT 0.0,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Seed some demo data on first run
    if is_new:
        seed = [
            ("HR26DK8337", "Joel George",  "9876543210", 45.00),
            ("DL04CC1234", "Alice Smith",   "9123456780", 12.50),
            ("KA05HG9988", "Bob Builder",   "9988776655", 100.00),
        ]
        c.executemany(
            "INSERT OR IGNORE INTO vehicles (plate, owner, phone, balance) VALUES (?, ?, ?, ?)",
            seed,
        )
        print(f"Seeded {len(seed)} demo vehicles into {DB_PATH}")
    conn.commit()

def db_lookup(plate):
    """Return dict with owner info if plate exists and is active, else None."""
    conn = _get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM vehicles WHERE plate = ? AND status = 'active'", (plate,))
    row = c.fetchone()
    if row:
        return {
            "owner": row["owner"],
            "phone": row["phone"],
            "balance": f"${row['balance']:.2f}",
            "balance_raw": row["balance"],
        }
    return None

def db_register(plate, owner, phone, initial_balance=0.0):
    """Insert a new vehicle. Returns True on success, False if already exists."""
    conn = _get_db()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO vehicles (plate, owner, phone, balance) VALUES (?, ?, ?, ?)",
            (plate, owner, phone, initial_balance),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def db_deduct_balance(plate, amount):
    """Deduct amount from vehicle balance. Returns new balance or None on failure."""
    conn = _get_db()
    c = conn.cursor()
    c.execute("SELECT balance FROM vehicles WHERE plate = ? AND status = 'active'", (plate,))
    row = c.fetchone()
    if row is None:
        return None
    new_balance = row["balance"] - amount
    if new_balance < 0:
        return None  # Insufficient funds
    c.execute("UPDATE vehicles SET balance = ? WHERE plate = ?", (new_balance, plate))
    conn.commit()
    return new_balance

def db_add_balance(plate, amount):
    """Add amount to vehicle balance. Returns new balance or None."""
    conn = _get_db()
    c = conn.cursor()
    c.execute("SELECT balance FROM vehicles WHERE plate = ? AND status = 'active'", (plate,))
    row = c.fetchone()
    if row is None:
        return None
    new_balance = row["balance"] + amount
    c.execute("UPDATE vehicles SET balance = ? WHERE plate = ?", (new_balance, plate))
    conn.commit()
    return new_balance

# ─── Plate Validation ────────────────────────────────

# Regex for License Plate (Generic Alphanumeric for now, can be specific like ^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$)
# Matches things like MH12DE1433, KA05AB1234
PLATE_PATTERN = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$')

def validate_plate(text):
    # simple cleanup
    text = text.replace(" ", "").upper()
    return PLATE_PATTERN.match(text) is not None, text

# ─── WebSocket Handler ───────────────────────────────

async def handle_connection(websocket):
    global _last_ocr_plate, _last_ocr_time, _latest_frame_jpg
    connected_clients.add(websocket)
    print(f"Client connected ({len(connected_clients)} total)")
    try:
        async for message in websocket:
            try:
                # ── JSON text message (registration / payment commands) ──
                if isinstance(message, str):
                    await handle_text_message(websocket, message)
                    continue

                # ── Frame-skipping: drain buffer and keep only the latest ──
                latest_frame = message
                try:
                    while True:
                        # non-blocking receive — grab any queued frames
                        next_msg = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                        if isinstance(next_msg, bytes):
                            latest_frame = next_msg   # discard old, keep newer
                        else:
                            # it's a text command that arrived between frames
                            await handle_text_message(websocket, next_msg)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass  # no more queued messages — proceed with latest_frame

                # ── Binary message (camera frame) ──
                loop = asyncio.get_running_loop()
                image = await loop.run_in_executor(None, decode_image, latest_frame)
                
                if image is None:
                    print("Failed to decode image")
                    continue

                # Run inference in thread pool
                boxes, scores, class_ids = await loop.run_in_executor(None, model.run_inference, image)
                
                # ── OCR cooldown: reuse cached result if recent ──
                now = time.monotonic()
                skip_ocr = (now - _last_ocr_time) < OCR_COOLDOWN_SECONDS and _last_ocr_plate is not None

                # Format results and run OCR
                results = []
                verified_data = None
                
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
                        if skip_ocr:
                            # Reuse cached OCR result
                            text_content = _last_ocr_plate
                            user_data = db_lookup(text_content)
                            if user_data:
                                verified_data = {
                                    "type": "verified",
                                    "plate": text_content,
                                    "owner": user_data["owner"],
                                    "balance": user_data["balance"]
                                }
                            else:
                                verified_data = {
                                    "type": "unregistered",
                                    "plate": text_content
                                }
                        else:
                            plate_crop = image[y1:y2, x1:x2]
                            # Run OCR on crop
                            ocr_result = await loop.run_in_executor(None, lambda: reader.readtext(plate_crop, detail=0))
                            
                            # Join standard output and clean
                            raw_text = "".join(ocr_result)
                            is_valid, clean_text = validate_plate(raw_text)
                                 
                            print(f"OCR Raw: '{raw_text}' -> Clean: '{clean_text}' | Valid Regex: {is_valid}")
                                 
                            if is_valid:
                                text_content = clean_text
                                _last_ocr_plate = clean_text
                                _last_ocr_time = now
                                # Check SQLite DB
                                user_data = db_lookup(clean_text)
                                if user_data:
                                    print(f"-> Found in DB: {clean_text}")
                                    verified_data = {
                                        "type": "verified",
                                        "plate": clean_text,
                                        "owner": user_data["owner"],
                                        "balance": user_data["balance"]
                                    }
                                else:
                                    print(f"-> NOT in DB (unregistered): {clean_text}")
                                    verified_data = {
                                        "type": "unregistered",
                                        "plate": clean_text
                                    }
                            else:
                                text_content = clean_text # Display even if invalid pattern
                    
                    results.append({
                        "class_id": int(class_id),
                        "score": float(score),
                        "box": box.tolist(),
                        "text": text_content
                    })

                # If we found a verified or unregistered plate, send that signal
                if verified_data:
                    print(f"PLATE RESULT: {verified_data}")
                    msg = json.dumps(verified_data)
                    await broadcast(msg)
                else:
                    # Send back standard JSON detections
                    msg = json.dumps(results)
                    await broadcast(msg)
                
                # Debugging: Log text to console
                print(f"Detected: {[r['text'] for r in results]}")
                
                # Fire-and-forget debug save — do NOT await
                loop.run_in_executor(None, save_debug_image, image, results)
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                import traceback
                traceback.print_exc()
                
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        connected_clients.discard(websocket)
        print(f"Client removed ({len(connected_clients)} remaining)")


async def broadcast(message):
    """Send a message to all connected WebSocket clients."""
    if connected_clients:
        await asyncio.gather(
            *[client.send(message) for client in connected_clients],
            return_exceptions=True
        )


async def handle_text_message(websocket, message):
    """Handle JSON commands from the client: register, payment, add_balance."""
    try:
        data = json.loads(message)
        action = data.get("action")

        if action == "register":
            plate = data.get("plate", "").strip().upper()
            owner = data.get("owner", "").strip()
            phone = data.get("phone", "").strip()
            initial_balance = float(data.get("initial_balance", 0.0))

            if not plate or not owner:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Plate and owner name are required."
                }))
                return

            success = db_register(plate, owner, phone, initial_balance)
            if success:
                print(f"REGISTERED: {plate} -> {owner}")
                await websocket.send(json.dumps({
                    "type": "registration_success",
                    "plate": plate,
                    "owner": owner,
                    "balance": f"${initial_balance:.2f}"
                }))
            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Plate {plate} is already registered."
                }))

        elif action == "payment":
            plate = data.get("plate", "").strip().upper()
            amount = float(data.get("amount", 0))

            if amount <= 0:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": "Invalid payment amount."
                }))
                return

            # Brief processing delay for UX feedback
            await asyncio.sleep(0.2)

            new_balance = db_add_balance(plate, amount)
            if new_balance is not None:
                print(f"PAYMENT: {plate} +${amount:.2f} -> new balance ${new_balance:.2f}")
                await websocket.send(json.dumps({
                    "type": "payment_success",
                    "plate": plate,
                    "amount": f"${amount:.2f}",
                    "new_balance": f"${new_balance:.2f}"
                }))
            else:
                await websocket.send(json.dumps({
                    "type": "error",
                    "message": f"Payment failed. Plate {plate} not found."
                }))

        else:
            print(f"Unknown action: {action}")
            await websocket.send(json.dumps({
                "type": "error",
                "message": f"Unknown action: {action}"
            }))

    except json.JSONDecodeError:
        print(f"Invalid JSON text message: {message[:100]}")
    except Exception as e:
        print(f"Error handling text message: {e}")
        await websocket.send(json.dumps({
            "type": "error",
            "message": str(e)
        }))


# ─── Utilities ───────────────────────────────────────

def save_debug_image(image, results):
    global _latest_frame_jpg
    try:
        debug_image = image.copy()
        for res in results:
            box = res["box"]
            text = res["text"]
            score = res["score"]
            
            x1, y1, x2, y2 = np.array(box).astype(int)
            
            # Draw box
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw text
            label = f"{text} ({score:.2f})" if text else f"{score:.2f}"
            cv2.putText(debug_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Store latest frame for HTTP feed
        _, jpg = cv2.imencode('.jpg', debug_image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        _latest_frame_jpg = jpg.tobytes()
        
        cv2.imwrite("latest_server_inference.jpg", debug_image)
    except Exception as e:
        print(f"Error saving debug image: {e}")

def decode_image(message):
    np_arr = np.frombuffer(message, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def get_network_ips():
    """Return likely LAN IP addresses for this machine."""
    ips = []

    # Preferred local IP via UDP route probe
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            route_ip = s.getsockname()[0]
            if route_ip and not route_ip.startswith("127."):
                ips.append(route_ip)
    except Exception:
        pass

    # Add additional interface IPs
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            ip = info[4][0]
            if ip and not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass

    return ips

# ─── HTTP Feed Server ─────────────────────────────────

async def handle_feed(request):
    """Serve the latest processed frame as JPEG."""
    if _latest_frame_jpg:
        return web.Response(body=_latest_frame_jpg, content_type='image/jpeg')
    return web.Response(status=204)

async def handle_feed_status(request):
    """Simple status endpoint."""
    return web.Response(
        text=json.dumps({"status": "ok", "clients": len(connected_clients)}),
        content_type='application/json'
    )

async def handle_api_users(request):
    """Return all active users from database as JSON."""
    conn = _get_db()
    c = conn.cursor()
    c.execute("SELECT plate, owner, balance FROM vehicles WHERE status = 'active'")
    rows = c.fetchall()
    users = [{"plate": r["plate"], "owner": r["owner"], "balance": f"${r['balance']:.2f}"} for r in rows]
    return web.Response(text=json.dumps(users), content_type='application/json')

import urllib.request
import urllib.error

def report_ip_to_vercel(endpoint_url="https://your-vercel-project.vercel.app/api/report-ip"):
    """Fetch the local IP and send it to a Vercel endpoint."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        
        data = json.dumps({"ip": local_ip, "timestamp": time.time()}).encode('utf-8')
        req = urllib.request.Request(endpoint_url, data=data, headers={'Content-Type': 'application/json'})
        
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                print(f"✅ Reported IP {local_ip} to Vercel (Status: {response.status})")
        except urllib.error.URLError as e:
            print(f"⚠️ Failed to report IP to Vercel: {e}")
            
    except Exception as e:
        print(f"⚠️ Error finding/reporting local IP: {e}")

# ─── Main ────────────────────────────────────────────

async def main(model_path, port):
    global model, reader
    
    # Report IP to Vercel
    report_ip_to_vercel()
    
    # Initialize SQLite database
    init_database()
    print(f"Database initialized at {DB_PATH}")
    
    print(f"Loading model from {model_path}...")
    model = YOLOInference(model_path)
    print("Model loaded.")
    
    print("Loading OCR reader...")
    reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if running on proper GPU machine
    print("OCR reader loaded.")
    
    http_port = port + 1  # HTTP feed on next port (e.g. 8766)
    
    print(f"Starting server on port {port} (WS) and {http_port} (HTTP feed)...")
    network_ips = get_network_ips()
    if network_ips:
        print("\n=== Pi Server Connection Info ===")
        print("Use one of these in app config:")
        for ip in network_ips:
            print(f"- Network IP: {ip}")
            print(f"  WebSocket URL: ws://{ip}:{port}")
            print(f"  Feed URL: http://{ip}:{http_port}/feed")
        print("===============================\n")
    else:
        print("Could not detect LAN IP automatically. Use ifconfig/ipconfig to find your network IP.")

    # Start HTTP server for feed
    app = web.Application()
    app.router.add_get('/feed', handle_feed)
    app.router.add_get('/status', handle_feed_status)
    app.router.add_get('/api/users', handle_api_users)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", http_port)
    await site.start()

    async with websockets.serve(handle_connection, "0.0.0.0", port):
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fixed_model.onnx", help="Path to ONNX model")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    args = parser.parse_args()
    
    asyncio.run(main(args.model, args.port))
