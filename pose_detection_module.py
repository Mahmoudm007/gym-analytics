import math
import cv2
import cv2 as cv
import mediapipe as mp
import time
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calc_joints(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    # get the angle between three points
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


cap = cv2.VideoCapture(0)
counter = 0
stage = None

# setup mediapipe instance
with mp_pose.Pose(min_detection_confidence= 0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (600, 400))
        # detect
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # the line that make the detection
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            angle = calc_joints(shoulder, elbow, wrist)
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),2, cv2.LINE_AA
                        )
            if angle > 160:
                stage = 'down'
            if angle < 50 and stage == 'down':
                stage = 'up'
                counter += 1
        except:
            pass

        cv2.rectangle(image, (0, 0), (225,73), (245, 117, 16), -1)
        cv2.putText(image, 'SETS', (15,12), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0),
                    1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10,60),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (255,255,255),
                    2, cv2.LINE_AA)

        # Rendering
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(240,117,60),thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(240,60,230),thickness=2, circle_radius=2)
                                  )


        cv2.imshow("Pose_detection", image)
        # Extract the pose
        h, w, c = image.shape
        opImg = np.zeros([h, w, c])
        opImg.fill(0)
        mp_drawing.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(240, 117, 60), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(240, 60, 230), thickness=2, circle_radius=2)
                                  )
        # cv2.imwrite("Extracted Image.jpg", opImg)
        cv2.imshow("Extracted position image", opImg)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

