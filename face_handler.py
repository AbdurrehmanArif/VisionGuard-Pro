import cv2
import os
import time
import streamlit as st
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from deepface import DeepFace

dataset_path = "dataset_faces"
os.makedirs(dataset_path, exist_ok=True)

def send_face_email(name, emp_id, sender, password, receiver):
    try:
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = receiver
        msg['Subject'] = '👤 Person Recognized'

        body = f"""
✅ PERSON RECOGNIZED

Name: {name}
Employee ID: {emp_id}
Time: {time.strftime("%Y-%m-%d %H:%M:%S")}

This person was just detected by the system.
"""
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print("Email error:", str(e))

def register_person(name, emp_id, cam_src=0):
    # Try with DirectShow on Windows if default fails or takes too long
    cap = cv2.VideoCapture(cam_src)
    if not cap.isOpened() and isinstance(cam_src, int):
        cap = cv2.VideoCapture(cam_src, cv2.CAP_DSHOW)

    st_frame = st.empty()
    status = st.empty()
    
    if not cap.isOpened():
        status.error(f"❌ Camera fail! Error opening camera {cam_src}. Close other tabs/scripts using it!")
        return

    person_dir = os.path.join(dataset_path, f"{name}_{emp_id}")
    os.makedirs(person_dir, exist_ok=True)
    
    count = 0
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    status.info("Recording video for exactly 10 seconds. Please look at the camera.")
    
    start_time = time.time()
    frames_read = 0
    last_save_time = 0
    
    while cap.isOpened() and (time.time() - start_time) < 10:
        ret, frame = cap.read()
        if not ret: 
            status.error("❌ Failed to read frame from camera! Please restart your camera or close other apps.")
            break
            
        frames_read += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Loosen detection parameters (1.1, 4) to ensure face is caught easily
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x,y,w,h) in faces:
            if time.time() - last_save_time >= 1.0:
                count += 1
                last_save_time = time.time()
                
                # Add margin to crop for better DeepFace alignment
                margin_x = int(w * 0.2)
                margin_y = int(h * 0.2)
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(frame.shape[1], x + w + margin_x)
                y2 = min(frame.shape[0], y + h + margin_y)
                
                face_img = frame[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(person_dir, f"face_{count}.jpg"), face_img)
            
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            break
            
        remaining_time = max(0, 10 - int(time.time() - start_time))
        cv2.putText(frame, f"Recording: {remaining_time}s | Saved: {count}/10", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", width="stretch")
        
    cap.release()
    
    if frames_read == 0:
         status.error(f"❌ Registration failed. No video frames received from camera {cam_src}. Camera might be in use.")
    elif count == 0:
         status.warning(f"⚠️ Video played, but 0 faces were detected. Please ensure your face is well lit and clearly visible.")
    else:
         status.success(f"✅ Registration complete for {name} ({emp_id})! {count} face images saved in 10 seconds.")
    
    pkl_file = os.path.join(dataset_path, "representations_vgg_face.pkl")
    if os.path.exists(pkl_file):
        os.remove(pkl_file)

def run_live_face_recognition(cam_src, email_enabled, sender, password, receiver):
    cap = cv2.VideoCapture(cam_src)
    st_frame = st.empty()
    log_area = st.empty()
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    last_match_name = "Unknown"
    last_check_time = 0
    emails_sent = set()
    
    logs = []
    
    while cap.isOpened() and st.session_state.get('camera_active_face', False):
        ret, frame = cap.read()
        if not ret: break
        
        current_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0 and (current_time - last_check_time) > 2.0:
            last_check_time = current_time
            
            # Check if there are any registered faces before running DeepFace
            if len(os.listdir(dataset_path)) > 0:
                try:
                    # Extract the first face crop with margin
                    x, y, w, h = faces[0]
                    margin_x = int(w * 0.2)
                    margin_y = int(h * 0.2)
                    x1 = max(0, x - margin_x)
                    y1 = max(0, y - margin_y)
                    x2 = min(frame.shape[1], x + w + margin_x)
                    y2 = min(frame.shape[0], y + h + margin_y)
                    face_crop = frame[y1:y2, x1:x2]
                    
                    # In-memory recognition to avoid file I/O errors
                    dfs = DeepFace.find(img_path=face_crop, db_path=dataset_path, enforce_detection=False, silent=True)
                    
                    if len(dfs) > 0 and len(dfs[0]) > 0:
                        best_match = dfs[0].iloc[0]
                        if best_match.get('distance', 1.0) <= 0.40:
                            matched_path = best_match['identity']
                            folder_name = os.path.basename(os.path.dirname(matched_path.replace("\\", "/")))
                            last_match_name = folder_name
                            logs.append(f"<span style='color:green'>Match: {folder_name}</span>")
                        else:
                            last_match_name = "Unknown"
                            logs.append(f"<span style='color:red'>Unknown Face (Dist: {best_match.get('distance', 1.0):.2f})</span>")
                    else:
                        last_match_name = "Unknown"
                        logs.append("<span style='color:red'>Unknown Face</span>")
                        
                        if email_enabled and last_match_name != "Unknown" and last_match_name not in emails_sent:
                            try:
                                name, emp_id = last_match_name.split('_')
                            except:
                                name, emp_id = last_match_name, "N/A"
                            send_face_email(name, emp_id, sender, password, receiver)
                            emails_sent.add(last_match_name)
                            logs.append(f"Email sent for {name} ({emp_id})")
                except Exception as e:
                    last_match_name = "Unknown"
            else:
                last_match_name = "No Data"

                
        for (x,y,w,h) in faces:
            color = (0,255,0) if last_match_name != "Unknown" and last_match_name != "No Data" else (255,0,0)
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
            cv2.putText(frame, last_match_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
        st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", width="stretch")
        if logs:
            log_area.markdown("<br>".join(logs), unsafe_allow_html=True)
            
    cap.release()

def get_person_identity(frame):
    try:
        if len(os.listdir(dataset_path)) == 0:
             return "No Data"
        # Explicitly extract the face using Haarcascade first
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        target_img = frame
        if len(faces) > 0:
            x, y, w, h = faces[0]
            margin_x = int(w * 0.2)
            margin_y = int(h * 0.2)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(frame.shape[1], x + w + margin_x)
            y2 = min(frame.shape[0], y + h + margin_y)
            target_img = frame[y1:y2, x1:x2]

        # In-memory recognition for background identification
        dfs = DeepFace.find(img_path=target_img, db_path=dataset_path, enforce_detection=False, silent=True)
        if len(dfs) > 0 and len(dfs[0]) > 0:
             best_match = dfs[0].iloc[0]
             # Default DeepFace threshold for VGG-Face (0.68) is too loose. 
             # We use 0.40 to be strict and accurately label unknown people.
             if best_match.get('distance', 1.0) <= 0.40:
                 matched_path = best_match['identity']
                 folder_name = os.path.basename(os.path.dirname(matched_path.replace("\\", "/")))
                 print(f"DEBUG: Found identity: {folder_name} (Dist: {best_match.get('distance', 1.0):.2f})")
                 return folder_name
             else:
                 print(f"DEBUG: Distance {best_match.get('distance', 1.0):.2f} > 0.40. Returning Unknown.")

        return "Unknown"

    except Exception as e:
        print("DeepFace Error:", e)
        return "Unknown"
