<div align="center">
  <h1>🚨 VisionGuard Pro: Advanced Distraction & Identity Monitoring System</h1>
  <p><i>A real-time Computer Vision system powered by YOLOv11, DeepFace, Streamlit, and MongoDB.</i></p>

  ![Python](https://img.shields.io/badge/Python-3.8--3.11-blue?style=for-the-badge&logo=python&logoColor=white)
  ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv&logoColor=white)
  ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit&logoColor=white)
  ![MongoDB](https://img.shields.io/badge/MongoDB-Database-brightgreen?style=for-the-badge&logo=mongodb&logoColor=white)
  ![YOLO](https://img.shields.io/badge/YOLOv11-Detection-yellow?style=for-the-badge)
</div>

---

## 📖 Overview

**VisionGuard Pro** is an enterprise-grade computer vision monitoring solution. It is specifically designed to detect physical distractions (such as mobile phone usage) in real-time. Unlike basic detection systems, VisionGuard goes a step further by **identifying the specific individual** committing the distraction using state-of-the-art facial recognition. 

When a prolonged distraction is detected, the system autonomously captures a screenshot, logs the event into a persistent MongoDB database, and dispatches an email alert with photographic evidence to administrators.

This project completely eliminates the need for manual surveillance in study halls, remote workspaces, or restricted secure zones.

---

## ✨ Core Features

| Feature | Description | Technology Used |
|---------|-------------|-----------------|
| **Real-Time Pose & Object Detection** | Tracks wrist coordinates and detects mobile devices. Calculates the Euclidean distance to determine if a phone is in active use. | Ultralytics YOLOv11 (Pose & Object) |
| **Facial Recognition** | Background thread processing identifies registered individuals dynamically without freezing the live camera feed. | DeepFace (VGG-Face), OpenCV |
| **Automated SMTP Alerts** | Sends instantaneous email alerts containing timestamps, culprit names, and attached image evidence when distraction time thresholds are breached. | Python `smtplib`, `email.mime` |
| **Persistent Event Logging** | All live stats, alert histories, and email logs are permanently stored in MongoDB — no data loss between sessions. | MongoDB, `pymongo` |
| **Analytics Dashboard** | A highly responsive, dark-themed UI with 8+ interactive charts, gauges, and statistical visualizations. | Streamlit, Plotly, Pandas |
| **Individual Search & Analysis** | Admin can search any registered person and view their personal violation history, charts, and photographic evidence. | Streamlit, Plotly |

---

## 📊 Dashboard Tabs

The dashboard is organized into **6 tabs** for complete monitoring coverage:

| Tab | Purpose |
|-----|---------|
| **📹 Live Detection** | Real-time camera feed with bounding boxes, wrist tracking, FPS counter, and live alert log. |
| **📊 Statistics** | 8 interactive Plotly charts analyzing distraction patterns across all registered persons. |
| **🔍 Search Person** | Search/select any individual to view their personal violation stats, timeline, and evidence screenshots. |
| **🖼️ Alert History** | Gallery of all captured violation screenshots with timestamps and download buttons. |
| **📧 Email Logs** | Complete record of every automated email sent — status, recipient, and timestamp. |
| **👤 Face Recognition** | Register new identities by recording 10 face crops via webcam in just 10 seconds. |

---

## 📈 Visualization Charts (Statistics Tab)

The **📊 Statistics** tab provides **8 premium Plotly charts** with a custom dark theme:

### Top-Level Analytics
| Chart | What It Shows |
|-------|---------------|
| **🎯 Event Distribution (Donut)** | Proportional breakdown of all logged events with vibrant color coding. |
| **👥 Most Distracted Individuals (Bar)** | Ranked bar chart showing who has the most distraction alerts — gradient from orange to red. |

### Time-Based Analytics & Severity
| Chart | What It Shows |
|-------|---------------|
| **Avg Duration Gauge (Speedometer)** | A 3-zone gauge (green/yellow/red) showing the average distraction duration with a delta arrow comparing against a 15-second baseline. |
| **📈 Cumulative Growth (Area)** | Spline area chart showing how alerts accumulate over time — reveals if distractions are accelerating. |

### Distribution
| Chart | What It Shows |
|-------|---------------|
| **📌 Duration Histogram** | Tri-color gradient histogram (green → orange → red) showing how often different distraction lengths occur. |

### Deep-Dive Analysis
| Chart | What It Shows |
|-------|---------------|
| **🔴 Severity Scatter** | Each alert as a bubble — size proportional to duration, color-coded by severity, with person name labels. |
| **⏱️ Total Time per Person (H-Bar)** | Horizontal bar showing total seconds each person spent distracted — not just count, but actual time. |
| **📦 Duration Box Plot** | Statistical box plot per person — shows min, max, median, quartiles, and outliers of distraction lengths. |
| **🌐 Person-Event Sunburst** | Inner ring = person, outer ring = event type — one glance shows the full breakdown. |

---

## 🔍 Individual Search Tab

Admins can select any registered person and instantly see:
- **3 Metric Cards:** Total alerts, screenshots captured, and overall status (✅ Clean / ⚠️ Warning / 🚨 Frequent Violator).
- **Distraction Duration Histogram:** Personal distribution of how long their distractions last.
- **Violation Timeline (Area Chart):** Cumulative growth of their violations over time.
- **Evidence Gallery:** Most recent screenshots captured during their violations.

---

## 🏗️ System Architecture Pipeline

1. **Frame Capture:** Camera captures the frame via OpenCV (`cv2.CAP_DSHOW` optimized for zero-latency buffering).
2. **Distraction Analysis (Every 5th Frame):** 
   - YOLOv11 detects phones.
   - YOLOv11-Pose maps human wrists.
   - If `distance(Phone_Center, Wrist) < Threshold`, the status becomes "Distracted".
3. **Identity Resolution:**
   - Haarcascade isolates the face from the YOLO bounding box (with a 20% margin for context).
   - DeepFace extracts embeddings and matches them against the registered database.
4. **Tolerance & Trigger Mechanism:**
   - A timer starts. If the distraction continues beyond the user-defined `Alert Time`, a trigger is fired.
5. **Database & Notification:**
   - Event is inserted into MongoDB (`vision_guard` database → `alerts` and `emails` collections).
   - `smtplib` packages the frame into an email and sends it to the admin.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have the following installed on your machine:
- **Python 3.8 to 3.11**
- **MongoDB Community Server**: Download from [here](https://www.mongodb.com/try/download/community). Ensure it is running locally on port `27017`.

### 2. Clone and Install Dependencies
```bash
git clone https://github.com/AbdurrehmanArif/VisionGuard-Pro.git
cd VisionGuard-Pro
pip install -r requirements.txt
```

### 3. Email Configuration (Gmail)
To allow the system to send emails, you must generate a Google App Password:
1. Go to your Google Account → Security → 2-Step Verification.
2. Scroll down to **App Passwords**.
3. Create a new App Password for "Mail".
4. *You can either enter this password directly into the Streamlit Sidebar during runtime or set it as a default inside `distraction_email.py`.*

---

## 🎮 Comprehensive Usage Guide

To launch the system, run the following command in your terminal:
```bash
streamlit run dashboard.py
```

### Phase 1: Registering Identities
Before running live detection, the system needs to know who to look for.
1. Navigate to the **"👤 Face Recognition"** tab on the dashboard.
2. Enter the target's **Name** and **Employee/Student ID**.
3. Click **Start Local Registration**.
4. The camera will activate. Look directly into the camera for 10 seconds. The system will intelligently throttle captures to save exactly 10 high-quality, aligned face crops to the `dataset_faces` directory.

### Phase 2: Live Monitoring
1. Switch to the **"📹 Live Detection"** tab.
2. Open the left **⚙️ Settings Sidebar**:
   - **Confidence & Wrist Distance:** Adjust how strict the AI should be.
   - **Alert Time & Tolerance:** Define how many seconds a person is allowed to hold their phone before it counts as a violation.
   - **Email Config:** Enter your credentials and toggle **"Enable Email Alerts"** ON.
3. Click **▶ Start Detection**. 
4. The system will draw bounding boxes around phones, map wrist vectors, and attach the registered name to the person dynamically.

### Phase 3: Reviewing Analytics
- **📊 Statistics Tab:** View 8 interactive charts analyzing distraction patterns, severity, and individual performance.
- **🔍 Search Person Tab:** Select any individual to view their personal violation history with charts and screenshot evidence.
- **🖼️ Alert History Tab:** Browse through a gallery of past violations. You can directly download the screenshot evidence from here.
- **📧 Email Logs Tab:** Check the status of every automated email sent to ensure no alerts failed due to network issues.

---

## 📁 Project Structure

```
VisionGuard-Pro/
├── dashboard.py              # Main Streamlit app (UI, detection loop, charts)
├── face_handler.py           # Face registration & DeepFace identification
├── distraction_email.py      # SMTP email configuration & standalone detection
├── destraction_detection.py  # Core YOLO detection logic
├── requirements.txt          # Python dependencies
├── .gitignore                # Excludes models, venv, screenshots
├── README.md                 # This file
├── dataset_faces/            # Registered face images (auto-created)
└── screenshots/              # Alert evidence images (auto-created)
```

---

## ⚡ Performance Optimizations

VisionGuard is heavily optimized to run efficiently on standard hardware without severe lag:
- **DirectShow Backend:** Prevents Windows OpenCV from queueing old frames, completely eliminating "Webcam Delay / Latency".
- **Throttled Inference:** YOLO models only execute once every 5 frames. Missing intermediate frames use cached bounding boxes to prevent visual flickering.
- **WebSocket Throttling:** Streamlit UI images are updated only every 2 frames, halving the network payload between the Python backend and the browser.
- **Contextual Face Cropping:** Faces are cropped with a `20% pixel margin`, drastically improving DeepFace's alignment accuracy and reducing false "Unknown" labels.
- **Threaded Face Recognition:** DeepFace matching runs in a background `ThreadPoolExecutor`, preventing the main video feed from freezing while the AI computes embeddings.

---

## 🐛 Troubleshooting & FAQ

**Q: The Live Camera is lagging or delayed by 3 seconds!**
> A: Ensure no other applications (like Zoom or Skype) are fighting for the camera. The system uses `cv2.CAP_DSHOW` by default to fix this on Windows. 

**Q: Face Recognition always returns "Unknown".**
> A: Clear the `dataset_faces` folder entirely. Ensure your room is well-lit, and re-register your face from the dashboard.

**Q: Streamlit shows a MongoDB Connection Error.**
> A: Your MongoDB background service is not running. On Windows, open "Services", find "MongoDB Server", and click "Start".

**Q: Emails are failing to send.**
> A: You cannot use your normal Gmail password. You *must* generate an App Password from Google Security settings.

**Q: Charts are empty in the Statistics tab.**
> A: Charts populate only after at least one distraction alert has been triggered. Start live detection and wait for an alert.

---

## 🛡️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend / UI** | Streamlit, Custom CSS (Inter font, dark theme) |
| **Computer Vision** | OpenCV, Ultralytics YOLOv11, DeepFace (VGG-Face) |
| **Data Visualization** | Plotly (8+ chart types), Pandas |
| **Database** | MongoDB (persistent alerts & email logs) |
| **Notifications** | Python `smtplib` (Gmail SMTP) |
| **Concurrency** | `ThreadPoolExecutor`, `concurrent.futures` |

---
<div align="center">
  <b>Developed by Abdurrehman Arif</b><br>
  <i>Seamless, automated, and intelligent workplace monitoring.</i>
</div>
