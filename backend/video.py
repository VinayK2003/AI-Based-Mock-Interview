import cv2
import numpy as np
import mediapipe as mp
import dlib
from scipy.spatial import distance as dist
import argparse
import math
import time
import os
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe Face Mesh for facial landmarks
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

# Load emotion detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_classifier = load_model('model.h5')  # Make sure this model file exists
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Constants 
EAR_THRESHOLD = 0.25 
NO_FACE_THRESHOLD = 90 
LONG_EYE_CLOSURE_THRESHOLD = 90
POSE_HOLD_THRESHOLD = 60 
NORMAL_BLINK_RATE = 15  
BLINK_RATE_PENALTY_THRESHOLD = NORMAL_BLINK_RATE * 2 
BLINK_RATE_PENALTY = 0.5 

def reset_metrics():
    """Reset all tracking metrics to initial state."""
    return {
        "blink_count": 0,
        "long_eye_closure_count": 0,
        "no_face_count": 0,
        "emotion_count": {
            "Angry": 0, "Disgust": 0, "Fear": 0, 
            "Happy": 0, "Neutral": 0, "Sad": 0, "Surprise": 0
        },
        "head_movement": {
            "left_look_count": 0,
            "right_look_count": 0,
            "up_look_count": 0,
            "down_look_count": 0,
            "left_lean_count": 0,
            "right_lean_count": 0
        }
    }

def calculate_additional_metrics(metrics, duration_seconds):
    # Blink rate calculation
    blink_rate = (metrics["blink_count"] / duration_seconds) * 60 if duration_seconds > 0 else 0
    
    # Calculate positive emotions percentage
    emotion_count = metrics["emotion_count"]
    positive_emotions = emotion_count["Happy"] + emotion_count["Surprise"] + emotion_count["Neutral"]
    total_emotions = sum(emotion_count.values())
    positive_percentage = (positive_emotions / total_emotions) * 100 if total_emotions > 0 else 0

    # Calculate looking/leaning away counts
    head_movement = metrics["head_movement"]
    total_looking_away = (
        head_movement["left_look_count"] + 
        head_movement["right_look_count"] + 
        head_movement["up_look_count"] + 
        head_movement["down_look_count"]
    )
    total_leaning_away = (
        head_movement["left_lean_count"] + 
        head_movement["right_lean_count"]
    )

    # Interview score calculation
    interview_score = 10.0
    
    # Emotion percentage deductions
    if positive_percentage >= 80:
        pass
    elif 70 <= positive_percentage < 80:
        interview_score -= 0.3
    elif 50 <= positive_percentage < 70:
        deduction = 0.5 + (0.2 * ((70 - positive_percentage) / 20))
        interview_score -= deduction
    else:
        interview_score -= 1.0
    
    # Other deductions
    interview_score -= metrics["long_eye_closure_count"] * 0.5
    interview_score -= metrics["no_face_count"] * 0.5    
    interview_score -= total_looking_away * 0.5    
    interview_score -= total_leaning_away * 0.3
    
    # Blink rate deduction
    if blink_rate > BLINK_RATE_PENALTY_THRESHOLD:
        interview_score -= BLINK_RATE_PENALTY

    interview_score = max(0, interview_score)

    return {
        "blink_rate": blink_rate,
        "positive_emotions_percentage": positive_percentage,
        "total_looking_away": total_looking_away,
        "total_leaning_away": total_leaning_away,
        "interview_score": round(interview_score, 2),
        "raw_metrics": metrics
    }

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calculate_face_angle(mesh):
    if not mesh:
        return {}

    # Extract required landmarks from MediaPipe Face Mesh
    nose_tip = mesh[4]
    forehead = mesh[10]
    left_eye_inner = mesh[133]
    right_eye_inner = mesh[362]
    chin = mesh[152]

    # Calculate vectors for orientation
    h_vec = np.array([right_eye_inner.x - left_eye_inner.x,
                     right_eye_inner.y - left_eye_inner.y,
                     right_eye_inner.z - left_eye_inner.z])
    
    v_vec = np.array([forehead.x - chin.x,
                     forehead.y - chin.y,
                     forehead.z - chin.z])
    
    # Calculate angles
    roll = np.arctan2(h_vec[1], h_vec[0])
    yaw = np.arctan2(h_vec[2], h_vec[0])
    pitch = np.arctan2(v_vec[2], v_vec[1])
    
    return {
        "roll": roll,
        "yaw": yaw,
        "pitch": pitch
    }

def classify_head_pose(angles):
    roll = angles["roll"]
    yaw = angles["yaw"]
    pitch = angles["pitch"]

    # Classify roll (leaning left/right)
    if roll < -0.1:
        roll_status = "Leaning Right"
    elif roll > 0.1:
        roll_status = "Leaning Left"
    else:
        roll_status = "Straight (No Lean)"

    # Classify yaw (turning left/right)
    if yaw < -0.1:
        yaw_status = "Looking Right"
    elif yaw > 0.1:
        yaw_status = "Looking Left"
    else:
        yaw_status = "Straight (No Turn)"

    # Classify pitch (looking up/down)
    if pitch < -0.1:
        pitch_status = "Looking Down"
    elif pitch > 0.1:
        pitch_status = "Looking Up"
    else:
        pitch_status = "Straight (No Tilt)"

    return roll_status, yaw_status, pitch_status

def classify_facial_expression(frame, face_rect):
    # Convert dlib rectangle to coordinates
    x, y, w, h = face_rect.left(), face_rect.top(), face_rect.width(), face_rect.height()
    
    # Extract ROI and preprocess
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi_gray = gray[y:y+h, x:x+w]
    
    try:
        # Resize to 48x48 as expected by most emotion recognition models
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        if np.sum([roi_gray]) == 0:
            return "No Face"
            
        # Normalize and prepare for prediction
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        # Make prediction
        prediction = emotion_classifier.predict(roi)[0]
        label = emotion_labels[prediction.argmax()]
        
        return label
        
    except Exception as e:
        print(f"Expression detection error: {e}")
        return "Neutral"

def analyze_video(video_path):
    # Reset metrics for each video analysis
    metrics = reset_metrics()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    start_time = time.time()
    total_frames = 0
    
    # Tracking variables
    cooldown_counter = 0
    eye_closed_frames = 0
    no_face_frames = 0
    current_pose = None
    pose_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1
        
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Detect facial landmarks using dlib
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 0:
            no_face_frames += 1
            if no_face_frames == NO_FACE_THRESHOLD:
                metrics["no_face_count"] += 1
        else:
            no_face_frames = 0  # Reset counter if face is detected

        for face in faces:
            landmarks = predictor(gray, face)

            # Extract eye landmarks for EAR calculation
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Detect eye blink
            if avg_ear < EAR_THRESHOLD:
                if cooldown_counter == 0:
                    metrics["blink_count"] += 1
                    cooldown_counter = 5  # Start cooldown
                eye_closed_frames += 1
                if eye_closed_frames == LONG_EYE_CLOSURE_THRESHOLD:
                    metrics["long_eye_closure_count"] += 1
            else:
                if eye_closed_frames > 0:
                    eye_closed_frames = 0  # Reset counter only when eyes are open

            # Decrement cooldown counter
            if cooldown_counter > 0:
                cooldown_counter -= 1

            # Classify facial expression
            expression = classify_facial_expression(frame, face)
            if expression in metrics["emotion_count"]:
                metrics["emotion_count"][expression] += 1
            
            # Calculate face angles using MediaPipe Face Mesh
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mesh = face_landmarks.landmark
                    angles = calculate_face_angle(mesh)
                    roll_status, yaw_status, pitch_status = classify_head_pose(angles)
                    
                    # Update head movement counters
                    if yaw_status == "Looking Left":
                        metrics["head_movement"]["left_look_count"] += 1
                    elif yaw_status == "Looking Right":
                        metrics["head_movement"]["right_look_count"] += 1
                    
                    if pitch_status == "Looking Up":
                        metrics["head_movement"]["up_look_count"] += 1
                    elif pitch_status == "Looking Down":
                        metrics["head_movement"]["down_look_count"] += 1

    cap.release()
    
    # Calculate duration
    duration_seconds = time.time() - start_time
    
    # Process final metrics
    final_analysis = calculate_additional_metrics(metrics, duration_seconds)
    
    return final_analysis

@app.route('/upload-video/<email>/<int:question_index>', methods=['POST'])
def upload_video(email, question_index):
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400
    
    video_file = request.files['video']
    
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
        video_file.save(temp_file.name)
        temp_file_path = temp_file.name
    
    try:
        # Analyze the video
        analysis_result = analyze_video(temp_file_path)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return jsonify({
            "analysis": analysis_result
        })
    
    except Exception as e:
        # Ensure temporary file is deleted even if an error occurs
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
    