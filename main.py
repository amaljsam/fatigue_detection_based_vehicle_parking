import cv2
import dlib
import numpy as np
from scipy.spatial import distance


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

EYE_AR_THRESHOLD = 0.25
EYE_AR_CONSEC_FRAMES = 20
MOUTH_AR_THRESHOLD = 0.7

frame_counter = 0
fatigue_alert = False

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        mouth = landmarks[48:68]

        ear_left = eye_aspect_ratio(left_eye)
        ear_right = eye_aspect_ratio(right_eye)
        ear = (ear_left + ear_right) / 2.0
        mar = mouth_aspect_ratio(mouth)

        for point in left_eye:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)  
        for point in right_eye:
            cv2.circle(frame, tuple(point), 2, (0, 255, 0), -1)  

        for point in mouth:
            cv2.circle(frame, tuple(point), 2, (255, 0, 0), -1)  

        if ear < EYE_AR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= EYE_AR_CONSEC_FRAMES:
                fatigue_alert = True
        else:
            frame_counter = 0
            fatigue_alert = False

        if mar > MOUTH_AR_THRESHOLD:
            fatigue_alert = True

        if fatigue_alert:
            cv2.putText(frame, "FATIGUE DETECTED! PARKING VEHICLE...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Fatigue Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
