from flask import Flask, render_template, Response, request, jsonify, send_from_directory
from detector import video_generator, global_metrics, stop_detection
import os
import time
import config

app = Flask(__name__, template_folder='.')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video")
def video_feed():
    source = request.args.get("src") or config.LIVE_FEED 
    return Response(video_generator(source),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/count")
def get_count():
    return jsonify(global_metrics)

@app.route("/api/snapshots")
def get_snapshots():
    try:
        snapshot_files = [f for f in os.listdir(config.SNAPSHOT_DIR) if f.endswith('.jpg')]
        snapshot_files.sort(key=lambda x: os.path.getmtime(os.path.join(config.SNAPSHOT_DIR, x)), reverse=True)
        return jsonify(snapshot_files[:10])
    except Exception:
        return jsonify([])

@app.route("/api/alerts")
def get_alerts():
    try:
        with open(config.LOG_FILE, 'r') as f:
            lines = f.readlines()
        
        alert_lines = [line.strip() for line in lines if "VIOLATION" in line]
        return jsonify(alert_lines[-10:])
    except Exception:
        return jsonify(["Log file inaccessible or no violations recorded."])
    
@app.route("/stop")
def stop_feed():
    stop_detection()
    return jsonify({"status": "Stopping detection. Please close the feed window."})

@app.route("/snapshots/<filename>")
def serve_snapshot(filename):
    return send_from_directory(config.SNAPSHOT_DIR, filename)

if __name__ == "__main__":
    if not os.path.exists(config.SNAPSHOT_DIR): os.makedirs(config.SNAPSHOT_DIR)
    if not os.path.exists(os.path.dirname(config.LOG_FILE)): os.makedirs(os.path.dirname(config.LOG_FILE))

    app.run(host="0.0.0.0", port=5000, debug=True)