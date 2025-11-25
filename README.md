# PPE Compliance System (YOLO/Flask)

This project implements a real-time smart system for monitoring Personal Protective Equipment (PPE) compliance, specifically hardhat usage, using computer vision. The system processes live video feeds or static videos, tracks unique individuals, and logs violations with corresponding visual snapshots.

It is built upon a simplified YOLO detection pipeline integrated with a Python Flask web server for a live dashboard view.

## Key Features

* **Real-time Detection:** YOLO model inference for Hardhat/No-Hardhat classification.
* **Unique Tracking:** Uses an IOU Tracker to maintain an accurate count of active, unique personnel and prevent double-counting.
* **Web Dashboard:** Displays the live video feed, real-time unique person count, recent violation alerts, and snapshot gallery.
* **System Integrity:** Logs all system activity and violations (with timestamped snapshots) for auditing.

---

## Dataset and Model Information

The model utilized in this project was trained using custom datasets focused on industrial safety and construction sites.

| Resource Type              | Name / Source                                                                                       | License                      |
| :------------------------- | :-------------------------------------------------------------------------------------------------- | :--------------------------- |
| **Training Dataset** | [PPE Detection Dataset (Hugging Face)](https://huggingface.co/datasets/HB1204/PPE_Detection)           | Custom (See original source) |
| **Model Version**    | [PPE Detection Roboflow Universe](https://universe.roboflow.com/himanshu-bharati/ppe_dectection-dtt4q) | **CC BY 4.0**          |

---

## Installation and Setup

### Prerequisites

1. Python 3.8+
2. NVIDIA GPU with CUDA installed (Recommended for real-time performance)

### Step 1: Clone the Repository

```bash
https://github.com/himanshu120400/PPE_Detection.git
cd PPE_Detection
```

### Step 2: Install Dependencies

All required Python libraries (Flask, Ultralytics, OpenCV) are listed in requirements.txt.

```
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
```

```
`# Install packages
pip install -r requirements.txt`
```

### Step 3: Run the Application

Execute the main application file.

```
python app.py
```

Which will start the Flask Server.

### Step 4: Access the Dashboard

Open your web browser and navigate to the local server address:`[http://127.0.0.1:5000/](http://127.0.0.1:5000/)`

Enter an RTSP stream URL or a local video file path (e.g., helmet1.mp4) to begin the real-time detection feed.
