import cv2
import time
import logging
import os
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import config
from collections import defaultdict
import numpy as np

class IOUTracker:
    def __init__(self, max_lost=5, iou_thresh=0.35):
        self.next_id = 0
        self.tracks = {}
        self.max_lost = max_lost
        self.iou_thresh = iou_thresh

    def _iou(self, a,b):
        ax1,ay1,ax2,ay2 = a
        bx1,by1,bx2,by2 = b
        ix1,iy1 = max(ax1,bx1), max(ay1,by1)
        ix2,iy2 = min(ax2,bx2), min(ay2,by2)
        iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
        inter = iw*ih
        areaA = max(1,(ax2-ax1)*(ay2-ay1))
        areaB = max(1,(bx2-bx1)*(by2-by1))
        union = areaA + areaB - inter
        return inter/union if union>0 else 0

    def update(self, dets):
        assigned = {}
        dets = list(dets)
        if len(self.tracks)==0:
            for d in dets:
                self.tracks[self.next_id] = {"bbox":d, "lost":0}
                assigned[self.next_id] = d
                self.next_id += 1
            return assigned
        tids = list(self.tracks.keys())
        boxes = [self.tracks[t]["bbox"] for t in tids]
        if len(dets)==0:
            for tid in tids:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"]>self.max_lost:
                    del self.tracks[tid]
            return {}
        iou_m = np.zeros((len(boxes), len(dets)))
        for i,b in enumerate(boxes):
            for j,d in enumerate(dets):
                iou_m[i,j] = self._iou(b,d)
        matched_det = set()
        matched_tid = set()
        for i in range(iou_m.shape[0]):
            if iou_m.shape[1]==0: break
            j = int(iou_m[i].argmax())
            if iou_m[i,j] >= self.iou_thresh:
                tid = tids[i]
                self.tracks[tid]["bbox"] = dets[j]
                self.tracks[tid]["lost"] = 0
                assigned[tid] = dets[j]
                matched_det.add(j)
                matched_tid.add(tid)
        for j,d in enumerate(dets):
            if j not in matched_det:
                self.tracks[self.next_id] = {"bbox":d, "lost":0}
                assigned[self.next_id] = d
                self.next_id += 1
        for tid in list(self.tracks.keys()):
            if tid not in matched_tid:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"]>self.max_lost:
                    del self.tracks[tid]
        return assigned

global_tracker = IOUTracker()
global_metrics = {"person_count": 0}
global_stop_flag = False

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename=config.LOG_FILE,
    format="%(asctime)s — %(levelname)s — %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

logger.info("Downloading/loading YOLO model.")
try:
    model_path = hf_hub_download(
        repo_id=config.MODEL_REPO,
        filename=config.MODEL_FILE
    )
    model = YOLO(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

def load_capture(source):
    logger.info(f"Opening video source: {source} via FFMPEG")
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        logger.warning("FFMPEG failed. Trying default capture method.")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error(f"Failed to open video source using any method: {source}")
            return None
    
    return cap

def save_snapshot(frame, annotated):
    if not config.SAVE_SNAPSHOTS:
        return
    os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
    filename = f"alert_{int(time.time())}.jpg"
    filepath = os.path.join(config.SNAPSHOT_DIR, filename)
    cv2.imwrite(filepath, annotated) 
    logger.warning(f"Snapshot saved: {filepath}")
    return filename

def stop_detection():
    global global_stop_flag
    global_stop_flag = True
    logger.info("STOP signal received.")

def video_generator(source):
    global global_metrics, global_tracker, global_stop_flag
    
    if model is None:
        logger.error("Model is not loaded. Cannot start detection.")
        blank_frame = np.zeros((480, 640, 3), np.uint8)
        ok, buffer = cv2.imencode('.jpg', blank_frame)
        if ok:
             yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    cap = load_capture(source)
    if cap is None:
        return

    global_tracker = IOUTracker()
    global_metrics["person_count"] = 0
    global_stop_flag = False
    last_alert_time = 0
    fps_last_time = time.time()
    fps_counter = 0

    while True:
        if global_stop_flag:
                logger.info("Detection loop gracefully stopped by user signal.")
                break
        try:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame not received! Attempting to reconnect or stream end.")
                if source.endswith(('.mp4', '.avi', '.mov', '.jpg', '.png')):
                    break 
                cap.release()
                time.sleep(2)
                cap = load_capture(source)
                if cap is None: break
                continue

            results = model(frame, conf=0.35, iou=0.4, verbose=False)
            annotated = results[0].plot()

            detection_boxes = []
            helmet = 0
            no_helmet = 0

            for r in results:
                for box in r.boxes:
                    label = r.names[int(box.cls)].lower()
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                    detection_boxes.append((x1, y1, x2, y2))
                    
                    if "no-hardhat" in label:
                        no_helmet += 1
                    elif "hardhat" in label:
                        helmet += 1

            tracked = global_tracker.update(detection_boxes)
            global_metrics["person_count"] = len(tracked)
            
            total = helmet + no_helmet

            if total > 0 and no_helmet > 0:
                now = time.time()
                if now - last_alert_time > config.ALERT_COOLDOWN:
                    logger.warning(f"VIOLATION (NO-HELMET): Unique Persons Tracked={global_metrics['person_count']}, Detections={total}")
                    save_snapshot(frame, annotated) 
                    last_alert_time = now

            fps_counter += 1
            if time.time() - fps_last_time >= 1:
                logger.info(f"FPS: {fps_counter}")
                fps_counter = 0
                fps_last_time = time.time()

            ok, buffer = cv2.imencode('.jpg', annotated)
            if not ok: continue
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            logger.error(f"STREAMING ERROR: {e}")
            break

    if cap is not None:
        cap.release()
    logger.info(f"Detection stream ended for source: {source}")