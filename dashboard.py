import streamlit as st
import cv2
import time
import os
import numpy as np
import smtplib
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from ultralytics import YOLO
import threading
import concurrent.futures
from streamlit.runtime.scriptrunner import add_script_run_ctx



# Global ThreadPool for Face Recognition to avoid spawning new threads and eating memory
face_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

from pymongo import MongoClient

# ============================================
# MONGODB CONFIG
# ============================================
try:
    mongo_client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    db = mongo_client["vision_guard"]
    alerts_col = db["alerts"]
    emails_col = db["emails"]
    mongo_client.admin.command('ping')
except Exception as e:
    st.error(f"MongoDB connection failed: {e}")
    alerts_col = None
    emails_col = None

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Distraction Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main { background-color: #0e1117; }
    
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3d);
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px 0;
    }
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #ffffff;
    }
    .metric-label {
        font-size: 13px;
        color: #8892a4;
        margin-top: 5px;
    }
    .alert-card {
        background: linear-gradient(135deg, #2d1515, #3d1a1a);
        border: 1px solid #ff4444;
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
    }
    .normal-card {
        background: linear-gradient(135deg, #0d2d1a, #0f3320);
        border: 1px solid #00cc44;
        border-radius: 10px;
        padding: 15px;
        margin: 8px 0;
    }
    .status-distracted {
        background: linear-gradient(90deg, #ff4444, #cc0000);
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        animation: pulse 1s infinite;
    }
    .status-normal {
        background: linear-gradient(90deg, #00cc44, #009933);
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
    }
    .status-table {
        background: linear-gradient(90deg, #ffaa00, #cc8800);
        color: white;
        padding: 15px 25px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
    }
    .sidebar-header {
        font-size: 18px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 10px;
    }
    .log-entry {
        background: #1a1d2e;
        border-left: 3px solid #ff4444;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 0 8px 8px 0;
        font-size: 13px;
    }
    footer { display: none !important; }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE INIT
# ============================================
if 'alert_log' not in st.session_state:
    if alerts_col is not None:
        st.session_state.alert_log = list(alerts_col.find({}, {"_id": 0}).sort("_id", 1))
    else:
        st.session_state.alert_log = []

if 'email_log' not in st.session_state:
    if emails_col is not None:
        st.session_state.email_log = list(emails_col.find({}, {"_id": 0}).sort("_id", 1))
    else:
        st.session_state.email_log = []

if 'total_alerts' not in st.session_state:
    st.session_state.total_alerts = len(st.session_state.alert_log)
if 'total_emails' not in st.session_state:
    st.session_state.total_emails = len([e for e in st.session_state.email_log if '✅' in e.get('status', '')])
if 'detection_start' not in st.session_state: st.session_state.detection_start = None
if 'last_seen'       not in st.session_state: st.session_state.last_seen       = None
if 'alert_triggered' not in st.session_state: st.session_state.alert_triggered = False
if 'camera_active'   not in st.session_state: st.session_state.camera_active   = False
if 'person_identities' not in st.session_state: st.session_state.person_identities = {} # track_id -> {name, last_check}


os.makedirs('screenshots', exist_ok=True)

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_models():
    pose  = YOLO('yolo11n-pose.pt')
    phone = YOLO('yolo11n.pt')
    return pose, phone

pose_model, phone_model = load_models()

MOBILE      = 67
LEFT_WRIST  = 9
RIGHT_WRIST = 10

# ============================================
# SMTP EMAIL
# ============================================
def send_email(screenshot_path, elapsed, sender, password, receiver, person_name="Unknown"):
    try:
        msg            = MIMEMultipart()
        msg['From']    = sender
        msg['To']      = receiver
        msg['Subject'] = '🚨 Distraction Alert!'

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = f"""
🚨 DISTRACTION ALERT!

📅 Time     : {now_str}
⏱ Duration  : {int(elapsed)} seconds
📱 Status   : Person using mobile detected
👤 Person   : {person_name}

Screenshot attached.
-- Distraction Detection Dashboard
        """
        msg.attach(MIMEText(body, 'plain'))

        with open(screenshot_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment',
                           filename=os.path.basename(screenshot_path))
            msg.attach(img)

        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        return True, "Email sent!"
    except Exception as e:
        return False, str(e)

# ============================================
# DETECTION HELPERS
# ============================================
def get_center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)

def distance(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def draw_box(frame, box, label, color):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1-25), (x1+w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

def run_detection(frame, conf_thresh, wrist_dist):
    # Use track() instead of call() to get stable IDs
    pose_results = pose_model.track(frame, conf=conf_thresh, persist=True, stream=True, imgsz=320, verbose=False)
    pose_res = list(pose_results)[0]
    
    phone_res = list(phone_model(frame, conf=conf_thresh-0.1, iou=0.3, stream=True, imgsz=320, verbose=False))[0]


    phones, wrists = [], []

    for box in phone_res.boxes:
        if int(box.cls) == MOBILE:
            bbox = box.xyxy[0].tolist()
            conf = float(box.conf)
            phones.append({'box': bbox, 'conf': conf})
            draw_box(frame, bbox, f"Phone {conf:.2f}", (255,165,0))

    if pose_res.keypoints is not None:
        for kps in pose_res.keypoints.xy:
            for idx in [LEFT_WRIST, RIGHT_WRIST]:
                kp = kps[idx]
                x, y = float(kp[0]), float(kp[1])
                if x > 0 and y > 0:
                    wrists.append((x, y))
                    cv2.circle(frame, (int(x), int(y)), 8, (0,255,255), -1)

    if 'person_identities' not in st.session_state:
        st.session_state.person_identities = {}


    # Identify Persons & Match Identities
    distracted_person_name = "Unknown"
    
    if pose_res.boxes is not None:
        for box in pose_res.boxes:
            bbox = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get track ID (default to 0 if no tracker yet)
            track_id = int(box.id[0]) if box.id is not None else 0
            
            # Init state for this ID
            if track_id not in st.session_state.person_identities:
                st.session_state.person_identities[track_id] = {"name": "Unknown", "last_check": 0}
            
            p_state = st.session_state.person_identities[track_id]
            now = time.time()
            
            # If name is Unknown, or it's been a while, try to identify
            if p_state["name"] == "Unknown" and (now - p_state["last_check"] > 4.0):
                person_crop = frame[max(0, y1):y2, max(0, x1):x2]
                if person_crop.size > 0:
                    st.session_state.person_identities[track_id]["last_check"] = now
                    def update_identity_cb(crop, target_id):
                        try:
                            # Add context so it can update session_state safely
                            from face_handler import get_person_identity
                            ident = get_person_identity(crop)
                            if ident != "Unknown":
                                st.session_state.person_identities[target_id]["name"] = ident
                        except Exception as e:
                            print(f"DEBUG: Multi-Identity error for ID {target_id}: {e}")
                    
                    # Submit with context
                    t = threading.Thread(target=update_identity_cb, args=(person_crop.copy(), track_id))
                    add_script_run_ctx(t)
                    t.start()

            # Draw per-person label with Name and System ID
            current_name = p_state["name"]
            # Clean display: Name (ID: track_id)
            display_label = f"👤 {current_name} (TRK:{track_id})"
            draw_box(frame, bbox, display_label, (0,255,0))
            
            # Save the name of the last seen person to potentially blame for distraction
            distracted_person_name = current_name



    draw_actions = []

    mobile_in_use = False
    for phone in phones:
        pc = get_center(phone['box'])
        for wrist in wrists:
            if distance(pc, wrist) < wrist_dist:
                mobile_in_use = True
                cv2.line(frame,
                         (int(wrist[0]), int(wrist[1])),
                         (int(pc[0]), int(pc[1])),
                         (0,0,255), 2)
                draw_actions.append(('line', (int(wrist[0]), int(wrist[1])), (int(pc[0]), int(pc[1])), (0,0,255), 2))
                draw_box(frame, phone['box'], "IN USE!", (0,0,255))
                draw_actions.append(('box', phone['box'], "IN USE!", (0,0,255)))

    # collect draw actions for things drawn earlier in the function
    for p in phones:
        draw_actions.append(('box', p['box'], f"Phone {p['conf']:.2f}", (255,165,0)))
    for w in wrists:
        draw_actions.append(('circle', (int(w[0]), int(w[1])), 8, (0,255,255), -1))
    if pose_res.boxes is not None:
        for box in pose_res.boxes:
            bbox = box.xyxy[0].tolist()
            track_id = int(box.id[0]) if box.id is not None else 0
            if track_id in st.session_state.person_identities:
                p_state = st.session_state.person_identities[track_id]
                display_label = f"👤 {p_state['name']} (TRK:{track_id})"
                draw_actions.append(('box', bbox, display_label, (0,255,0)))

    return frame, mobile_in_use, len(phones), len(pose_res.boxes) if pose_res.boxes is not None else 0, distracted_person_name, draw_actions


# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Settings</div>', unsafe_allow_html=True)

    st.subheader("🎯 Detection")
    conf_thresh = st.slider("Confidence",    0.1, 0.9, 0.5, 0.05)
    wrist_dist  = st.slider("Wrist Distance", 50, 300, 150, 10)
    alert_time  = st.slider("Alert Time (sec)", 10, 300, 120, 10)
    tolerance   = st.slider("Tolerance (sec)",   1,  30,   7,  1)

    st.divider()

    st.subheader("📧 Email Config")
    try:
        from distraction_email import EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER
    except ImportError:
        EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER = "", "", ""

    email_enabled  = st.toggle("Enable Email Alerts", value=False)
    email_sender   = st.text_input("Sender Email",   value=EMAIL_SENDER, placeholder="your@gmail.com")
    email_password = st.text_input("App Password",   value=EMAIL_PASSWORD, type="password")
    email_receiver = st.text_input("Receiver Email", value=EMAIL_RECEIVER, placeholder="receiver@gmail.com")

    if st.button("🧪 Test Email"):
        if email_sender and email_password and email_receiver:
            with st.spinner("Sending..."):
                ok, msg = send_email(
                    list(filter(lambda f: f.endswith('.jpg'),
                         os.listdir('screenshots') or ['']))[:1][0]
                    if os.listdir('screenshots') else 'screenshots',
                    0, email_sender, email_password, email_receiver
                )
            st.success("✅ Sent!") if ok else st.error(f"❌ {msg}")
        else:
            st.warning("Email details please!")

    st.divider()

    if st.button("🗑️ Clear All Logs", type="secondary"):
        st.session_state.alert_log    = []
        st.session_state.email_log    = []
        st.session_state.total_alerts = 0
        st.session_state.total_emails = 0
        if alerts_col is not None:
            alerts_col.delete_many({})
        if emails_col is not None:
            emails_col.delete_many({})
        st.rerun()

# ============================================
# MAIN DASHBOARD
# ============================================
st.title("🚨 Distraction Detection Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.divider()

# ---- TABS ----
tab1, tab2, tab_search, tab3, tab4, tab5 = st.tabs([
    "📹 Live Detection",
    "📊 Statistics",
    "🔍 Search Person",
    "🖼️ Alert History",
    "📧 Email Logs",
    "👤 Face Recognition"
])

# ============================================
# TAB 1 — LIVE DETECTION
# ============================================
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Camera Feed")
        cam_source = st.text_input("Camera Source", value="0",
                                   help="0=webcam, ya CCTV URL")

        c1, c2 = st.columns(2)
        start_btn = c1.button("▶ Start Detection", type="primary",  width="stretch")
        stop_btn  = c2.button("⏹ Stop",            type="secondary", width="stretch")

        frame_placeholder  = st.empty()
        status_placeholder = st.empty()
        timer_placeholder  = st.empty()

    with col2:
        st.subheader("📊 Live Stats")
        m1 = st.empty()
        m2 = st.empty()
        m3 = st.empty()
        m4 = st.empty()

        st.divider()
        st.subheader("📋 Live Log")
        log_placeholder = st.empty()

    # Detection loop
    if start_btn:
        st.session_state.camera_active   = True
        st.session_state.detection_start = None
        st.session_state.last_seen       = None
        st.session_state.alert_triggered = False
        st.session_state.person_identities = {} # Reset IDs on start


    if stop_btn:
        st.session_state.camera_active = False

    if st.session_state.camera_active:
        src = int(cam_source) if cam_source.isdigit() else cam_source
        
        # Windows DirectShow backend handles webcam buffering correctly without lag
        if isinstance(src, int) and os.name == 'nt':
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(src)
            
        cap.set(cv2.CAP_PROP_FPS,          30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        frame_idx = 0
        mobile_in_use = False
        phone_count = 0
        person_count = 0
        detected_name = "Unknown"
        
        fps_counter = 0
        fps_time = time.time()
        fps_display = 0
        
        last_draw_actions = []

        while st.session_state.camera_active:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera not open!")
                break

            frame = cv2.resize(frame, (640, 480))
            
            # PERFORMANCE FIX: Only run heavy AI every 5 frames to prevent processing lag
            if frame_idx % 5 == 0:
                frame, mobile_in_use, phone_count, person_count, detected_name, last_draw_actions = run_detection(
                    frame, conf_thresh, wrist_dist
                )
            else:
                # On intermediate frames, draw the cached boxes to avoid flickering
                for action in last_draw_actions:
                    if action[0] == 'box':
                        draw_box(frame, action[1], action[2], action[3])
                    elif action[0] == 'circle':
                        cv2.circle(frame, action[1], action[2], action[3], action[4])
                    elif action[0] == 'line':
                        cv2.line(frame, action[1], action[2], action[3], action[4])
                
            frame_idx += 1



            now = time.time()

            # Timer logic
            if mobile_in_use:
                st.session_state.last_seen = now
                if st.session_state.detection_start is None:
                    st.session_state.detection_start = now
                    st.session_state.alert_triggered = False
                    st.session_state.alert_log.append({
                        'time': datetime.now().strftime("%H:%M:%S"),
                        'event': 'Detection Started',
                        'duration': 0
                    })
            else:
                if st.session_state.last_seen is not None:
                    if (now - st.session_state.last_seen) > tolerance:
                        st.session_state.detection_start = None
                        st.session_state.last_seen       = None
                        st.session_state.alert_triggered = False

            # Alert check
            if (st.session_state.detection_start is not None
                    and not st.session_state.alert_triggered):
                elapsed   = now - st.session_state.detection_start
                remaining = alert_time - elapsed

                if elapsed >= alert_time:
                    st.session_state.alert_triggered = True
                    st.session_state.total_alerts   += 1

                    person_name = detected_name
                    # Clean filename for insurance
                    safe_name = person_name.replace(" ", "_").replace("/", "_")
                    filename = f"screenshots/alert_{safe_name}_{int(now)}.jpg"
                    cv2.imwrite(filename, frame)


                    alert_doc = {
                        'time':     datetime.now().strftime("%H:%M:%S"),
                        'event':    f'🚨 ALERT: {person_name}',
                        'duration': int(elapsed),
                        'person':   person_name,
                        'file':     filename
                    }
                    st.session_state.alert_log.append(alert_doc)
                    if alerts_col is not None:
                        alerts_col.insert_one(alert_doc.copy())


                    if email_enabled and email_sender and email_password and email_receiver:
                        ok, msg = send_email(filename, elapsed,
                                             email_sender, email_password, email_receiver,
                                             person_name)

                        email_doc = {
                            'time':   datetime.now().strftime("%H:%M:%S"),
                            'to':     email_receiver,
                            'status': '✅ Sent' if ok else f'❌ {msg}',
                            'file':   os.path.basename(filename)
                        }
                        st.session_state.email_log.append(email_doc)
                        if emails_col is not None:
                            emails_col.insert_one(email_doc.copy())
                        if ok: st.session_state.total_emails += 1

                    st.session_state.detection_start = None
                    st.session_state.last_seen       = None

                # Timer on frame
                mins = int(remaining) // 60
                secs = int(remaining) % 60
                cv2.rectangle(frame, (0,0), (frame.shape[1], 55), (0,0,200), -1)
                cv2.putText(frame, f"DISTRACTED! Alert in: {mins:02d}:{secs:02d}",
                            (15,38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)

                progress = min(elapsed / alert_time, 1.0)
                bar_w    = int(frame.shape[1] * progress)
                cv2.rectangle(frame, (0, frame.shape[0]-15),
                              (bar_w, frame.shape[0]), (0,0,255), -1)

                status_placeholder.markdown(
                    '<div class="status-distracted">🚨 DISTRACTED!</div>',
                    unsafe_allow_html=True)
                timer_placeholder.progress(progress, text=f"Alert in {mins:02d}:{secs:02d}")

            elif mobile_in_use is False and phone_count > 0:
                cv2.rectangle(frame, (0,0), (frame.shape[1], 55), (0,200,200), -1)
                cv2.putText(frame, "Phone on TABLE", (15,38),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
                status_placeholder.markdown(
                    '<div class="status-table">📱 Phone on Table — OK</div>',
                    unsafe_allow_html=True)
                timer_placeholder.empty()
            else:
                cv2.rectangle(frame, (0,0), (frame.shape[1], 55), (0,150,0), -1)
                cv2.putText(frame, "Normal", (15,38),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
                status_placeholder.markdown(
                    '<div class="status-normal">✅ Normal</div>',
                    unsafe_allow_html=True)
                timer_placeholder.empty()

            # FPS
            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_time    = time.time()

            cv2.putText(frame, f"FPS: {fps_display}",
                        (frame.shape[1]-120, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            # Update UI elements only every 2 frames to reduce Streamlit WebSocket lag
            if frame_idx % 2 == 0:
                # Show frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", width="stretch")

                # Live stats
                m1.markdown(f'<div class="metric-card"><div class="metric-value">{person_count}</div><div class="metric-label">👤 Persons</div></div>', unsafe_allow_html=True)
                m2.markdown(f'<div class="metric-card"><div class="metric-value">{phone_count}</div><div class="metric-label">📱 Phones</div></div>', unsafe_allow_html=True)
                m3.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_alerts}</div><div class="metric-label">🚨 Total Alerts</div></div>', unsafe_allow_html=True)
                m4.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_emails}</div><div class="metric-label">📧 Emails Sent</div></div>', unsafe_allow_html=True)

                # Live log
                if st.session_state.alert_log:
                    log_html = ""
                    for entry in st.session_state.alert_log[-5:][::-1]:
                        log_html += f'<div class="log-entry">⏰ {entry["time"]} — {entry["event"]}</div>'
                    log_placeholder.markdown(log_html, unsafe_allow_html=True)

        cap.release()

# ============================================
# TAB 2 — STATISTICS
# ============================================
with tab2:
    st.subheader("📊 Detection Statistics")

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_alerts}</div><div class="metric-label">🚨 Total Alerts</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_emails}</div><div class="metric-label">📧 Emails Sent</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-value">{len(os.listdir("screenshots"))}</div><div class="metric-label">📸 Screenshots</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><div class="metric-value">{len(st.session_state.alert_log)}</div><div class="metric-label">📋 Log Entries</div></div>', unsafe_allow_html=True)

    st.divider()

    df = pd.DataFrame(st.session_state.alert_log) if st.session_state.alert_log else pd.DataFrame()
    alerts_only = df[df['event'].str.startswith('🚨 ALERT')] if not df.empty else pd.DataFrame()

    st.markdown("### 📇 Registered Persons Directory")
    registered_people = []
    if os.path.exists("dataset_faces"):
        registered_people = [d for d in os.listdir("dataset_faces") if os.path.isdir(os.path.join("dataset_faces", d))]
    
    db_records = []
    for person in registered_people:
        alerts = 0
        if not alerts_only.empty and 'person' in alerts_only.columns:
            alerts = len(alerts_only[alerts_only['person'] == person])
        
        status = "🚨 Frequent Violator" if alerts > 3 else ("⚠️ Warning" if alerts > 0 else "✅ Clean Record")
        db_records.append({
            "Registered Name (Emp ID)": person.replace("_", " - "),
            "Total Distractions": alerts,
            "Status": status
        })
        
    if db_records:
        db_df = pd.DataFrame(db_records)
        st.dataframe(db_df.sort_values(by="Total Distractions", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("No persons registered in the database yet.")

    st.divider()

    if not df.empty:

        CHART_LAYOUT = dict(
            paper_bgcolor='rgba(14,17,23,0.95)',
            plot_bgcolor='rgba(30,33,48,0.7)',
            font=dict(family='Inter, sans-serif', size=13, color='#c9d1d9'),
            title_font=dict(size=17, color='#ffffff', family='Inter, sans-serif'),
            margin=dict(l=20, r=20, t=50, b=30),
            hoverlabel=dict(bgcolor='#1e2130', font_size=13, font_family='Inter'),
        )

        st.markdown("### 📈 Top-Level Analytics")
        col1, col2 = st.columns(2)

        with col1:
            events = df['event'].value_counts().reset_index()
            fig = px.pie(events, values='count', names='event',
                         title='🎯 Event Distribution',
                         hole=0.55,
                         color_discrete_sequence=['#ff4b4b','#00e5ff','#ff9f43','#26de81','#a55eea'])
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label',
                textfont_size=12,
                marker=dict(line=dict(color='#0e1117', width=3))
            )
            fig.update_layout(**CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if not alerts_only.empty and 'person' in df.columns:
                person_counts = alerts_only['person'].value_counts().reset_index()
                person_counts.columns = ['Person', 'Distraction Count']
                fig3 = go.Figure(go.Bar(
                    x=person_counts['Person'],
                    y=person_counts['Distraction Count'],
                    text=person_counts['Distraction Count'],
                    textposition='outside',
                    marker=dict(
                        color=person_counts['Distraction Count'],
                        colorscale=[[0, '#ff9f43'], [0.5, '#ee5a24'], [1.0, '#ff0000']],
                        line=dict(color='rgba(255,75,75,0.4)', width=1.5)
                    ),
                    hovertemplate="<b>%{x}</b><br>Distractions: %{y}<extra></extra>"
                ))
                fig3.update_layout(
                    title='👥 Most Distracted Individuals',
                    xaxis_tickangle=-20,
                    yaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
                    **CHART_LAYOUT
                )
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("No person distraction data available yet.")

        st.divider()
        st.markdown("### ⏱️ Time-Based Analytics & Severity")
        col3, col4 = st.columns(2)

        with col3:
            if not alerts_only.empty:
                avg_duration = alerts_only['duration'].mean()
                max_range = max(60, int(alerts_only['duration'].max()) + 10)
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=avg_duration,
                    delta={'reference': 15, 'increasing': {'color': '#ff4b4b'}, 'decreasing': {'color': '#26de81'}},
                    title={'text': "Avg Distraction Duration (sec)", 'font': {'size': 16, 'color': '#ffffff'}},
                    number={'font': {'color': '#ff4b4b', 'size': 42}},
                    gauge={
                        'axis': {'range': [0, max_range], 'tickwidth': 1, 'tickcolor': '#555'},
                        'bar': {'color': '#ff4b4b', 'thickness': 0.25},
                        'bgcolor': 'rgba(0,0,0,0)',
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, max_range*0.33], 'color': 'rgba(38,222,129,0.15)'},
                            {'range': [max_range*0.33, max_range*0.66], 'color': 'rgba(255,159,67,0.15)'},
                            {'range': [max_range*0.66, max_range], 'color': 'rgba(255,75,75,0.15)'},
                        ],
                        'threshold': {'line': {'color': '#ff0000', 'width': 4}, 'thickness': 0.8, 'value': 30}
                    }
                ))
                fig_gauge.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig_gauge, use_container_width=True)

        with col4:
            if not alerts_only.empty:
                alerts_timeline = alerts_only.sort_values(by='time').copy()
                alerts_timeline['Cumulative Alerts'] = range(1, len(alerts_timeline) + 1)
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=alerts_timeline['time'],
                    y=alerts_timeline['Cumulative Alerts'],
                    mode='lines+markers',
                    name='Cumulative Alerts',
                    fill='tozeroy',
                    fillcolor='rgba(255,75,75,0.15)',
                    line=dict(color='#ff4b4b', width=3, shape='spline'),
                    marker=dict(size=8, color='#ff9f43', line=dict(width=2, color='white')),
                    hovertemplate="Time: <b>%{x}</b><br>Total Alerts: <b>%{y}</b><extra></extra>"
                ))
                fig4.update_layout(
                    title='📈 Cumulative Distraction Growth',
                    yaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
                    xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
                    **CHART_LAYOUT
                )
                st.plotly_chart(fig4, use_container_width=True)

        st.divider()
        st.markdown("### 📊 Distraction Length Distribution")
        if not alerts_only.empty:
            fig_hist = go.Figure(go.Histogram(
                x=alerts_only['duration'],
                nbinsx=10,
                name='Distractions',
                marker=dict(
                    color=alerts_only['duration'],
                    colorscale=[[0,'#26de81'],[0.5,'#ff9f43'],[1,'#ff4b4b']],
                    line=dict(color='rgba(255,255,255,0.1)', width=1)
                ),
                hovertemplate="Duration: <b>%{x}s</b><br>Count: <b>%{y}</b><extra></extra>"
            ))
            fig_hist.update_layout(
                title='📌 Frequency of Distraction Durations',
                bargap=0.15,
                yaxis=dict(gridcolor='rgba(255,255,255,0.06)', title='Count'),
                xaxis=dict(title='Duration (Seconds)'),
                **CHART_LAYOUT
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    else:
        st.info("NO DATA FOUND — Start the detection to collect data!")

# ============================================
# TAB SEARCH — INDIVIDUAL ANALYSIS
# ============================================
with tab_search:
    st.subheader("🔍 Individual Person Analysis")
    
    registered_people = []
    if os.path.exists("dataset_faces"):
        registered_people = [d for d in os.listdir("dataset_faces") if os.path.isdir(os.path.join("dataset_faces", d))]
    
    if not registered_people:
        st.warning("No persons registered in the database yet.")
    else:
        search_query = st.selectbox("Select or Search for a Person:", ["-- Select Person --"] + registered_people)
        
        if search_query != "-- Select Person --":
            st.markdown(f"### Analysis for: **{search_query.replace('_', ' - ')}**")
            
            df = pd.DataFrame(st.session_state.alert_log) if st.session_state.alert_log else pd.DataFrame()
            person_alerts = pd.DataFrame()
            if not df.empty and 'person' in df.columns:
                person_alerts = df[(df['event'].str.startswith('🚨 ALERT')) & (df['person'] == search_query)]
            
            c1, c2, c3 = st.columns(3)
            total_distractions = len(person_alerts)
            
            person_screenshots = [f for f in os.listdir('screenshots') if search_query in f and f.endswith('.jpg')] if os.path.exists('screenshots') else []
            
            status = "🚨 Frequent Violator" if total_distractions > 3 else ("⚠️ Warning" if total_distractions > 0 else "✅ Clean Record")
            
            c1.markdown(f'<div class="metric-card"><div class="metric-value">{total_distractions}</div><div class="metric-label">Total Alerts</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><div class="metric-value">{len(person_screenshots)}</div><div class="metric-label">Screenshots Captured</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><div class="metric-value" style="font-size:24px; padding-top:10px;">{status}</div><div class="metric-label">Overall Status</div></div>', unsafe_allow_html=True)
            
            st.divider()
            
            if total_distractions > 0:
                PERSON_LAYOUT = dict(
                    paper_bgcolor='rgba(14,17,23,0.95)',
                    plot_bgcolor='rgba(30,33,48,0.7)',
                    font=dict(family='Inter, sans-serif', size=13, color='#c9d1d9'),
                    title_font=dict(size=16, color='#ffffff'),
                    margin=dict(l=20, r=20, t=50, b=30),
                    hoverlabel=dict(bgcolor='#1e2130', font_size=13)
                )
                col1, col2 = st.columns(2)

                with col1:
                    fig_dur = go.Figure(go.Histogram(
                        x=person_alerts['duration'],
                        nbinsx=5,
                        marker=dict(
                            color=person_alerts['duration'],
                            colorscale=[[0, '#f9ca24'], [0.5, '#f0932b'], [1, '#eb4d4b']],
                            line=dict(color='rgba(255,255,255,0.1)', width=1)
                        ),
                        hovertemplate="Duration: <b>%{x}s</b><br>Count: <b>%{y}</b><extra></extra>"
                    ))
                    fig_dur.update_layout(
                        title='⏱️ Distraction Durations',
                        bargap=0.1,
                        yaxis=dict(gridcolor='rgba(255,255,255,0.06)', title='Occurrences'),
                        xaxis=dict(title='Duration (sec)'),
                        **PERSON_LAYOUT
                    )
                    st.plotly_chart(fig_dur, use_container_width=True)

                with col2:
                    pa = person_alerts.sort_values(by='time').copy()
                    pa['Cumulative Alerts'] = range(1, len(pa) + 1)
                    fig_time = go.Figure()
                    fig_time.add_trace(go.Scatter(
                        x=pa['time'], y=pa['Cumulative Alerts'],
                        mode='lines+markers',
                        fill='tozeroy',
                        fillcolor='rgba(255,75,75,0.12)',
                        line=dict(color='#ff4b4b', width=3, shape='spline'),
                        marker=dict(size=9, color='#f9ca24', symbol='circle',
                                    line=dict(width=2, color='white')),
                        hovertemplate="Time: <b>%{x}</b><br>Total: <b>%{y}</b><extra></extra>"
                    ))
                    fig_time.update_layout(
                        title='📅 Violation Timeline',
                        yaxis=dict(gridcolor='rgba(255,255,255,0.06)'),
                        xaxis=dict(gridcolor='rgba(255,255,255,0.04)'),
                        **PERSON_LAYOUT
                    )
                    st.plotly_chart(fig_time, use_container_width=True)

                st.markdown("#### 🖼️ Evidence (Recent Screenshots)")
                if person_screenshots:
                    person_screenshots.sort(reverse=True)
                    cols = st.columns(min(3, len(person_screenshots)))
                    for j, col in enumerate(cols):
                        if j < len(person_screenshots):
                            fname = person_screenshots[j]
                            fpath = os.path.join('screenshots', fname)
                            with col:
                                st.image(Image.open(fpath), caption=fname, use_container_width=True)
            else:
                st.success(f"🎉 **{search_query.replace('_', ' ')}** has no recorded distractions! Clean record.")

# ============================================
# TAB 3 — ALERT HISTORY
# ============================================
with tab3:
    st.subheader("🖼️ Alert Screenshots")

    screenshots = sorted(
        [f for f in os.listdir('screenshots') if f.endswith('.jpg')],
        reverse=True
    )

    if screenshots:
        cols_per_row = 3
        for i in range(0, len(screenshots), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                if i+j < len(screenshots):
                    fname = screenshots[i+j]
                    fpath = os.path.join('screenshots', fname)
                    with col:
                        img = Image.open(fpath)
                        st.image(img, caption=fname, width="stretch")
                        ts = fname.replace('alert_','').replace('.jpg','')
                        try:
                            dt = datetime.fromtimestamp(int(ts))
                            st.caption(f"📅 {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                        except:
                            pass
                        with open(fpath, 'rb') as f:
                            col.download_button(
                                "⬇️ Download",
                                f.read(),
                                fname,
                                "image/jpeg",
                                width="stretch"
                            )
    else:
        st.info("NO screenshort here — know detection open!")

# ============================================
# TAB 4 — EMAIL LOGS
# ============================================
with tab4:
    st.subheader("📧 Email Logs")

    if st.session_state.email_log:
        df_email = pd.DataFrame(st.session_state.email_log)
        st.dataframe(
            df_email,
            width="stretch",
            column_config={
                'time':   'Time',
                'to':     'Sent To',
                'status': 'Status',
                'file':   'Screenshot'
            }
        )

        sent   = len([e for e in st.session_state.email_log if '✅' in e['status']])
        failed = len([e for e in st.session_state.email_log if '❌' in e['status']])

        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#00cc44">{sent}</div><div class="metric-label">✅ Sent</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value" style="color:#ff4444">{failed}</div><div class="metric-label">❌ Failed</div></div>', unsafe_allow_html=True)
    else:
        st.info("NO EMAIL FOUND !")

# ============================================
# TAB 5 — FACE RECOGNITION
# ============================================
with tab5:
    st.subheader("👤 Face Recognition System")
    face_mode = st.radio("Select Mode:", ["Register New Person", "Live Detection"])
    
    if face_mode == "Register New Person":
        st.write("Register yourself by looking at the camera.")
        reg_name = st.text_input("Name (e.g. John)")
        reg_id = st.text_input("Employee/Person ID (e.g. 101)")
        cam_reg = st.text_input("Camera Source", value="0", key="cam_reg")
        if st.button("Start Local Registration"):
            if reg_name and reg_id:
                from face_handler import register_person
                src = int(cam_reg) if cam_reg.isdigit() else cam_reg
                register_person(reg_name, reg_id, src)
            else:
                st.warning("Please enter Name and ID to proceed.")
                
    elif face_mode == "Live Detection":
        st.write("Live detection of registered persons. Emails sent upon recognition.")
        cam_live = st.text_input("Camera Source", value="0", key="cam_live")
        c1, c2 = st.columns(2)
        start_face = c1.button("▶ Start Face Detection")
        stop_face = c2.button("⏹ Stop Face Detection")
        
        if 'camera_active_face' not in st.session_state:
            st.session_state.camera_active_face = False
            
        if start_face:
            st.session_state.camera_active_face = True
        if stop_face:
            st.session_state.camera_active_face = False
            
        if st.session_state.camera_active_face:
            from face_handler import run_live_face_recognition
            src = int(cam_live) if cam_live.isdigit() else cam_live
            run_live_face_recognition(src, email_enabled, email_sender, email_password, email_receiver)