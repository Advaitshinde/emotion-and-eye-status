import cv2
import cvzone
import numpy as np
import time
import base64
from flask import Flask, render_template, request, jsonify
from cvzone.FaceMeshModule import FaceMeshDetector

app = Flask(__name__)

# Global variables for tracking
blinkRatioList = []
blinkCounter = 0
counter = 0
color = (255, 0, 255)
BLINK_THRESHOLD = 32
last_blink_time = time.time()
blink_frequency = 0
steady_duration = 0
ratioAvg = 0
adaptive_threshold = BLINK_THRESHOLD

# Variables for tracking blink duration
blink_start_time = 0
blink_in_progress = False
blink_durations = []  # List to store individual blink durations
avg_blink_duration = 0.0  # Average blink duration in seconds
time_between_blinks = 0.0  # Time between consecutive blinks
last_complete_blink_time = time.time()  # Time of the last complete blink

# Eye landmarks
LEFT_EYE = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
RIGHT_EYE = [257, 258, 259, 260, 385, 386, 387, 388, 362, 263, 373, 374]

# Initialize detector
detector = FaceMeshDetector(maxFaces=1, minDetectionCon=0.7, minTrackCon=0.7)

# Reset function to clear global variables
def reset_variables():
    global blinkRatioList, blinkCounter, counter, last_blink_time, blink_frequency
    global steady_duration, ratioAvg, adaptive_threshold
    global blink_start_time, blink_in_progress, blink_durations
    global avg_blink_duration, time_between_blinks, last_complete_blink_time

    blinkRatioList = []
    blinkCounter = 0
    counter = 0
    color = (255, 0, 255)
    last_blink_time = time.time()
    blink_frequency = 0
    steady_duration = 0
    ratioAvg = 0
    adaptive_threshold = BLINK_THRESHOLD

    blink_start_time = 0
    blink_in_progress = False
    blink_durations = []
    avg_blink_duration = 0.0
    time_between_blinks = 0.0
    last_complete_blink_time = time.time()

def calculate_eye_metrics(face):
    # Get left eye measurements
    leftUp = face[159]
    leftDown = face[23]
    leftLeft = face[130]
    leftRight = face[243]
    
    # Calculate left eye distances
    lengthVer, _ = detector.findDistance(leftUp, leftDown)
    lengthHor, _ = detector.findDistance(leftLeft, leftRight)
    
    # Get right eye measurements
    rightUp = face[386]
    rightDown = face[374]
    rightLeft = face[362]
    rightRight = face[263]
    
    # Calculate right eye distances
    rightLengthVer, _ = detector.findDistance(rightUp, rightDown)
    rightLengthHor, _ = detector.findDistance(rightLeft, rightRight)
    
    # Calculate eye ratios
    if lengthHor > 0 and rightLengthHor > 0:
        leftRatio = int((lengthVer / lengthHor) * 100)
        rightRatio = int((rightLengthVer / rightLengthHor) * 100)
        ratio = (leftRatio + rightRatio) / 2
        return ratio, leftRatio, rightRatio
    return 0, 0, 0

def determine_eye_status(ratio, adaptive_threshold):
    if ratio < adaptive_threshold:
        return "Closed"
    elif ratio > 45:
        return "Fully Open"
    else:
        return "Partially Open"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    global blinkRatioList, blinkCounter, counter, last_blink_time, blink_frequency
    global steady_duration, ratioAvg, adaptive_threshold
    global blink_start_time, blink_in_progress, blink_durations
    global avg_blink_duration, time_between_blinks, last_complete_blink_time

    # Reset variables for each analysis
    reset_variables()
    
    try:
        # Decode base64 image
        data = request.json['image']
        img_data = base64.b64decode(data.split(',')[1])
        np_img = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Detect face and landmarks
        image, faces = detector.findFaceMesh(image, draw=False)

        # Initialize eye status variables
        left_eye_status = "Unknown"
        right_eye_status = "Unknown"

        if faces:
            face = faces[0]
            
            # Calculate eye metrics
            ratio, leftRatio, rightRatio = calculate_eye_metrics(face)
            
            # Apply smoothing with moving average
            blinkRatioList.append(ratio)
            if len(blinkRatioList) > 5:  # Use last 5 frames for smoothing
                blinkRatioList.pop(0)
            ratioAvg = sum(blinkRatioList) / len(blinkRatioList)
            
            # Determine eye status
            left_eye_status = determine_eye_status(leftRatio, adaptive_threshold)
            right_eye_status = determine_eye_status(rightRatio, adaptive_threshold)
            
            # Blink detection logic
            if ratioAvg < adaptive_threshold:
                current_time = time.time()
                
                # Track blink start time
                if not blink_in_progress:
                    blink_start_time = current_time
                    blink_in_progress = True
                
                # Only count as blink if counter is 0 (prevent double counting)
                if counter == 0:
                    # Only count as blink if sufficient time has passed
                    if current_time - last_blink_time > 0.2:  # 200ms minimum between blinks
                        blinkCounter += 1
                        
                        # Update last complete blink time for calculating time between blinks
                        time_between_blinks = current_time - last_complete_blink_time
                        last_complete_blink_time = current_time
                        last_blink_time = current_time
                        
                        # Calculate blink frequency (blinks per minute)
                        if blinkCounter > 1:
                            elapsed_time = current_time - last_complete_blink_time
                            if elapsed_time > 0:
                                blink_frequency = (blinkCounter / elapsed_time) * 60
                
                counter = 1
            else:
                # If eyes are open and we were tracking a blink
                if blink_in_progress:
                    # Calculate blink duration
                    blink_duration = time.time() - blink_start_time
                    
                    # Only count reasonable blink durations (between 0.1 and 0.5 seconds)
                    if 0.1 <= blink_duration <= 0.5:
                        blink_durations.append(blink_duration)
                        
                        # Calculate average blink duration
                        if blink_durations:
                            avg_blink_duration = sum(blink_durations) / len(blink_durations)
                    
                    # Reset blink tracking
                    blink_in_progress = False
            
            # Reset counter for next blink
            if counter != 0:
                counter += 1
                if counter > 7:
                    counter = 0

        return jsonify({
            'blinks': blinkCounter,
            'blink_rate': blink_frequency,
            'left_eye_status': left_eye_status,
            'right_eye_status': right_eye_status,
            'avg_blink_duration': avg_blink_duration,
            'time_between_blinks': time_between_blinks
        })

    except Exception as e:
        print(f"Error in analysis: {e}")
        return jsonify({
            'blinks': 0,
            'blink_rate': 0,
            'left_eye_status': 'Unknown',
            'right_eye_status': 'Unknown',
            'avg_blink_duration': 0,
            'time_between_blinks': 0,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')