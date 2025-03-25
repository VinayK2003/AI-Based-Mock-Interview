from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn
import cv2
import numpy as np
import tempfile
import os
import mediapipe as mp
import math
from datetime import datetime

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices
LEFT_EYE_LANDMARKS = [362, 263, 386, 374, 385, 380]  # Outer points of left eye
RIGHT_EYE_LANDMARKS = [33, 133, 160, 158, 144, 153]   # Outer points of right eye
LEFT_IRIS_LANDMARKS = [468, 469, 470, 471, 472]      # Left iris landmarks
RIGHT_IRIS_LANDMARKS = [473, 474, 475, 476, 477]     # Right iris landmarks

class EyeMovementResult(BaseModel):
    left_count: int
    right_count: int
    center_count: int
    total_frames: int
    metrics: dict

@app.get("/health-check")
async def health_check():
    return {"status": "online"}

@app.post("/upload-video/{user_email}/{question_index}")
async def upload_video(
    user_email: str,
    question_index: int,
    video: UploadFile = File(...),
    transcription: Optional[str] = Form(None)
):
    print(f"\n====== Processing video for user: {user_email}, question: {question_index} ======")
    print(f"Received video file: {video.filename}, size: {video.size} bytes")
    if transcription:
        print(f"Transcription length: {len(transcription)} characters")
    
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
        temp_file_path = temp_file.name
        contents = await video.read()
        temp_file.write(contents)
        print(f"Saved temporary video to: {temp_file_path}")

    try:
        # Process the video to track eye movements
        print("Starting eye movement analysis...")
        results = analyze_eye_movements(temp_file_path)
        
        # Print detailed results
        print("\n=== EYE MOVEMENT ANALYSIS RESULTS ===")
        print(f"Left movements:  {results['left_count']}")
        print(f"Right movements: {results['right_count']}")
        print(f"Center focus:    {results['center_count']}")
        print(f"Total frames:    {results['total_frames']}")
        print("===================================")
        
        # Additional metrics
        attention_score = calculate_attention_score(results)
        eye_movement_ratio = calculate_eye_movement_ratio(results)
        wpm = estimate_wpm(transcription) if transcription else None
        filler_words = count_filler_words(transcription) if transcription else 0
        
        print("\n=== CALCULATED METRICS ===")
        print(f"Attention score:      {attention_score:.2f}")
        print(f"Eye movement ratio:   {eye_movement_ratio:.2f}")
        if wpm:
            print(f"Estimated speech rate: {wpm} WPM")
        if transcription:
            print(f"Filler word count:     {filler_words}")
        print("=========================")
        
        # Combine results with transcription and metrics
        response_data = {
            "success": True,
            "user_email": user_email,
            "question_index": question_index,
            "eye_movements": results,
            "metrics": {
                "wpm": wpm,
                "clarity": 0.85,  # Placeholder - would need audio analysis
                "confidence": 0.75,  # Placeholder - would need deeper analysis
                "filler_words": filler_words,
                "eye_movement_ratio": eye_movement_ratio,
                "attention_score": attention_score
            }
        }
        
        print("\nAnalysis complete, returning results to frontend")
        return response_data
    
    except Exception as e:
        print(f"\nERROR: Video processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing failed: {str(e)}")
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Removed temporary file: {temp_file_path}")

def analyze_eye_movements(video_path):
    """
    Analyze eye movements from a video file.
    Returns counts of left, right, and center eye movements.
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise Exception("Could not open video file")
    
    # Get video properties for debugging
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video properties: {total_frames} frames, {fps:.2f} FPS, {duration:.2f} seconds")
    
    frame_count = 0
    processed_count = 0
    left_count = 0
    right_count = 0
    center_count = 0
    
    # For tracking movement (to avoid counting small jitters)
    prev_direction = "center"
    direction_buffer = []
    buffer_size = 5  # Number of frames to consider for a stable direction
    
    # For real-time feedback during processing
    progress_interval = max(1, total_frames // 10)
    
    print("Starting frame-by-frame analysis...")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        
        # Print progress
        if frame_count % progress_interval == 0:
            percent_done = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Processing: {percent_done:.1f}% complete ({frame_count}/{total_frames} frames)")
        
        # Skip frames for performance (process every 3rd frame)
        if frame_count % 3 != 0:
            continue
            
        processed_count += 1
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame with MediaPipe
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Get iris and eye landmarks
            landmarks = face_landmarks.landmark
            
            # Get eye direction based on iris position relative to eye corners
            direction = determine_eye_direction(landmarks, frame.shape)
            
            # Add to buffer for smoothing
            direction_buffer.append(direction)
            if len(direction_buffer) > buffer_size:
                direction_buffer.pop(0)
            
            # Only count if we have enough stable readings
            if len(direction_buffer) == buffer_size:
                # Check if the last N frames have the same direction
                if all(d == direction_buffer[0] for d in direction_buffer):
                    current_stable_direction = direction_buffer[0]
                    
                    # Only count if direction has changed
                    if current_stable_direction != prev_direction:
                        if current_stable_direction == "left":
                            left_count += 1
                            print(f"Frame {frame_count}: Detected LEFT eye movement")
                        elif current_stable_direction == "right":
                            right_count += 1
                            print(f"Frame {frame_count}: Detected RIGHT eye movement")
                        elif current_stable_direction == "center":
                            center_count += 1
                            print(f"Frame {frame_count}: Detected CENTER eye focus")
                        
                        prev_direction = current_stable_direction
    
    cap.release()
    
    print(f"\nProcessed {processed_count} frames out of {frame_count} total frames")
    print(f"Detected movements: {left_count} left, {right_count} right, {center_count} center")
    
    return {
        "left_count": left_count,
        "right_count": right_count,
        "center_count": center_count,
        "total_frames": frame_count,
        "processed_frames": processed_count
    }

def determine_eye_direction(landmarks, frame_shape):
    """
    Determine eye gaze direction based on iris position relative to eye corners.
    Returns "left", "right", or "center".
    """
    # Get landmarks for left and right iris
    left_iris = np.mean([(landmarks[idx].x, landmarks[idx].y) for idx in LEFT_IRIS_LANDMARKS], axis=0)
    right_iris = np.mean([(landmarks[idx].x, landmarks[idx].y) for idx in RIGHT_IRIS_LANDMARKS], axis=0)
    
    # Get eye corner landmarks
    left_eye_left = (landmarks[LEFT_EYE_LANDMARKS[0]].x, landmarks[LEFT_EYE_LANDMARKS[0]].y)
    left_eye_right = (landmarks[LEFT_EYE_LANDMARKS[3]].x, landmarks[LEFT_EYE_LANDMARKS[3]].y)
    
    right_eye_left = (landmarks[RIGHT_EYE_LANDMARKS[0]].x, landmarks[RIGHT_EYE_LANDMARKS[0]].y)
    right_eye_right = (landmarks[RIGHT_EYE_LANDMARKS[3]].x, landmarks[RIGHT_EYE_LANDMARKS[3]].y)
    
    # Calculate relative positions
    left_eye_width = left_eye_right[0] - left_eye_left[0]
    left_eye_iris_rel_pos = (left_iris[0] - left_eye_left[0]) / left_eye_width if left_eye_width > 0 else 0.5
    
    right_eye_width = right_eye_right[0] - right_eye_left[0]
    right_eye_iris_rel_pos = (right_iris[0] - right_eye_left[0]) / right_eye_width if right_eye_width > 0 else 0.5
    
    # Average both eyes
    avg_rel_pos = (left_eye_iris_rel_pos + right_eye_iris_rel_pos) / 2
    
    # Determine direction
    if avg_rel_pos < 0.45:  # Looking left
        return "left"
    elif avg_rel_pos > 0.55:  # Looking right
        return "right"
    else:  # Looking center
        return "center"

def calculate_attention_score(results):
    """Calculate an attention score based on eye movements"""
    total_movements = results["left_count"] + results["right_count"] + results["center_count"]
    center_ratio = results["center_count"] / max(total_movements, 1)
    
    # Higher center focus generally indicates better attention
    attention_score = (center_ratio * 0.7) + (0.3 * (1.0 - (results["left_count"] + results["right_count"]) / max(results["total_frames"], 1)))
    
    return min(max(attention_score, 0), 1)  # Scale between 0 and 1

def calculate_eye_movement_ratio(results):
    """Calculate ratio of eye movements to total frames"""
    total_movements = results["left_count"] + results["right_count"] + results["center_count"]
    return total_movements / max(results["processed_frames"], 1)  # Adjust for processed frames

def estimate_wpm(transcription):
    """Estimate words per minute from transcription"""
    if not transcription:
        return 0
        
    word_count = len(transcription.split())
    # Assuming average answer time of 60 seconds
    estimated_wpm = word_count
    return estimated_wpm

def count_filler_words(transcription):
    """Count filler words in transcription"""
    if not transcription:
        return 0
        
    filler_words = ["um", "uh", "like", "you know", "sort of", "kind of", "basically"]
    count = 0
    words = transcription.lower().split()
    
    for filler in filler_words:
        if " " in filler:  # Multi-word fillers
            count += transcription.lower().count(filler)
        else:  # Single-word fillers
            count += words.count(filler)
            
    return count

if __name__ == "__main__":
    print("\n=================================================================")
    print("Starting Eye Movement Analysis Server")
    print("This server will analyze video recordings and track eye movements")
    print("=================================================================\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)