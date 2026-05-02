from flask import Flask, render_template_string, send_file, Response
import os

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ethereal Charge - Admin Dashboard</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --border: #334155;
            --success: #22c55e;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .header {
            width: 100%;
            background-color: var(--card-bg);
            padding: 20px 40px;
            box-sizing: border-box;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            margin: 0;
            font-size: 24px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .header h1 span {
            color: var(--accent);
        }
        .status-badge {
            background-color: rgba(34, 197, 94, 0.2);
            color: var(--success);
            padding: 6px 12px;
            border-radius: 9999px;
            font-size: 14px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }
        .status-badge::before {
            content: '';
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: var(--success);
            border-radius: 50%;
        }
        .container {
            max-width: 1400px;
            width: 100%;
            padding: 40px;
            box-sizing: border-box;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        .card {
            background-color: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            display: flex;
            flex-direction: column;
        }
        .card-header {
            margin-bottom: 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .card-header h2 {
            margin: 0;
            font-size: 18px;
            color: var(--text-main);
            font-weight: 600;
        }
        .feed-container {
            width: 100%;
            aspect-ratio: 16/9;
            background-color: #000;
            border-radius: 12px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        #camera-feed {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .logs-container {
            background-color: #000;
            border-radius: 12px;
            padding: 16px;
            height: calc(100% - 44px);
            overflow-y: auto;
            font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
            font-size: 13px;
            line-height: 1.5;
            color: #a3be8c;
            white-space: pre-wrap;
            border: 1px solid #1a1a1a;
        }
        .log-line { border-bottom: 1px solid rgba(255,255,255,0.05); padding: 2px 0; }
        .empty-state {
            color: var(--text-muted);
            font-style: italic;
        }
        @media (max-width: 1024px) {
            .container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1><span>Ethereal Charge</span> Admin Dashboard</h1>
        <div class="status-badge">System Online</div>
    </div>
    
    <div class="container">
        <div class="card">
            <div class="card-header">
                <h2>Live Inference Feed</h2>
                <div style="font-size: 12px; color: var(--text-muted);">Updates 5fps</div>
            </div>
            <div class="feed-container">
                <img id="camera-feed" src="/feed" alt="Awaiting camera feed..." onerror="this.src='/placeholder'">
            </div>
        </div>
        
        <div class="card">
            <div class="card-header">
                <h2>Server Logs</h2>
                <div style="font-size: 12px; color: var(--text-muted);">Auto-scrolling</div>
            </div>
            <div class="logs-container" id="logs">
                <div class="empty-state">Loading logs...</div>
            </div>
        </div>
    </div>

    <script>
        // Update image feed
        const feedImg = document.getElementById('camera-feed');
        setInterval(() => {
            // Append timestamp to bust cache
            feedImg.src = '/feed?t=' + new Date().getTime();
        }, 200); // 5 FPS refresh rate

        // Fetch logs
        const logsContainer = document.getElementById('logs');
        let isScrolledToBottom = true;

        logsContainer.addEventListener('scroll', () => {
            isScrolledToBottom = logsContainer.scrollHeight - logsContainer.clientHeight <= logsContainer.scrollTop + 50;
        });

        async function fetchLogs() {
            try {
                const response = await fetch('/logs');
                const text = await response.text();
                if (text.trim() === "") {
                    logsContainer.innerHTML = '<div class="empty-state">No logs available (server.log is empty or missing).<br>Make sure you pipe the pi_server.py output to server.log</div>';
                } else {
                    const formatted = text.split('\\n').map(line => `<div class="log-line">${line}</div>`).join('');
                    logsContainer.innerHTML = formatted;
                    if (isScrolledToBottom) {
                        logsContainer.scrollTop = logsContainer.scrollHeight;
                    }
                }
            } catch (e) {
                console.error("Failed to fetch logs");
            }
        }

        setInterval(fetchLogs, 1000);
        fetchLogs();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/feed')
def feed():
    image_path = 'latest_server_inference.jpg'
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/jpeg')
    return Response("No image yet", status=404)

@app.route('/placeholder')
def placeholder():
    # Return a 1x1 transparent gif
    return Response(b'GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\x00\x00\x00!\xf9\x04\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02D\x01\x00;', mimetype='image/gif')

@app.route('/logs')
def logs():
    log_path = 'server.log'
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            # Get last 100 lines for performance
            lines = f.readlines()
            return "".join(lines[-100:])
    return ""

if __name__ == '__main__':
    print("Starting Ethereal Charge Admin Dashboard...")
    print("Access the dashboard at http://localhost:5000")
    print("Ensure pi_server.py is running and outputting to server.log (e.g. python3 pi_server.py | tee server.log)")
    app.run(host='0.0.0.0', port=5000)
