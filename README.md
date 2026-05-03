<div align="center">
  <h1>🚨 VisionGuard: Advanced Distraction & Identity Monitoring System</h1>
  <p><i>A real-time Computer Vision system powered by YOLOv11, DeepFace, Streamlit, and MongoDB.</i></p>
</div>

---

## 📖 Overview

**VisionGuard** is an enterprise-grade computer vision monitoring solution. It is specifically designed to detect physical distractions (such as mobile phone usage) in real-time. Unlike basic detection systems, VisionGuard goes a step further by **identifying the specific individual** committing the distraction using state-of-the-art facial recognition. 

When a prolonged distraction is detected, the system autonomously captures a screenshot, logs the event into a persistent database, and dispatches an email alert with photographic evidence to administrators.

This project completely eliminates the need for manual surveillance in study halls, remote workspaces, or restricted secure zones.

---

## ✨ Core Features

| Feature | Description | Technology Used |
|---------|-------------|-----------------|
| **Real-Time Pose & Object Detection** | Tracks wrist coordinates and detects mobile devices. Calculates the Euclidean distance to determine if a phone is in active use. | Ultralytics YOLOv11 (Pose & Object) |
| **Facial Recognition** | Background thread processing identifies registered individuals dynamically without freezing the live camera feed. | DeepFace (VGG-Face), OpenCV |
| **Automated SMTP Alerts** | Sends instantaneous email alerts containing timestamps, culprit names, and attached image evidence when distraction time thresholds are breached. | Python `smtplib`, `email.mime` |
| **Persistent Event Logging** | All live stats, alert histories, and email logs are permanently and securely stored rather than relying on volatile RAM. | MongoDB, `pymongo` |
| **Analytics Dashboard** | A highly responsive, dark-themed UI that visualizes live logs, calculates FPS, and renders interactive statistical charts. | Streamlit, Plotly, Pandas |

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
   - Event is inserted into MongoDB.
   - `smtplib` packages the frame into an email and sends it to the admin.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
Ensure you have the following installed on your machine:
- **Python 3.8 to 3.11**
- **MongoDB Community Server**: Download from [here](https://www.mongodb.com/try/download/community). Ensure it is running locally on port `27017`.

### 2. Clone and Install Dependencies
```bash
git clone <your-repo-url>
cd Computer-Vision-mobile-detection
pip install -r requirements.txt
```

### 3. Email Configuration (Gmail)
To allow the system to send emails, you must generate a Google App Password:
1. Go to your Google Account -> Security -> 2-Step Verification.
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
- **📊 Statistics Tab:** View interactive Pie charts and Bar graphs showing distraction frequencies and event distributions.
- **🖼️ Alert History Tab:** Browse through a gallery of past violations. You can directly download the screenshot evidence from here.
- **📧 Email Logs Tab:** Check the status of every automated email sent to ensure no alerts failed due to network issues.

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

---
<div align="center">
  <i>Developed for seamless, automated, and intelligent monitoring.</i>
</div>
