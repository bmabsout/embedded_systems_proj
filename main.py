import time

import cv2
import numpy as np
from imutils.video import VideoStream
from imutils import face_utils
import face_recognition as fr
import dlib
import imutils
import pose
import send_message as sm

# Set initial frame size.
frameSize = (640, 640)

# Initialize mutithreading the video stream.
vs = VideoStream(src=0, usePiCamera=True, resolution=frameSize,
                framerate=24).start()


from cascade_detector import face_detector

lower_purple = np.array([110,30,30])
upper_purple = np.array([140,255,255])

previous_angles = np.array([0,0,0])
smoothing = 0.9
current_setPoint = np.array([0,0,0])
smoothed_angles = np.array([0,0,0])
face_setPoint = np.array([0,0])
while True:
        # Get the next frame.
        frame = vs.read()
        if frame is None:
            continue
        frame = cv2.flip(frame, 1)
        #frame = imutils.resize(frame, height=500, width=500)
        # Show video stream
        #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #mask = cv2.inRange(hsv, lower_purple, upper_purple)
        #frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #rects = runCascade(gray, faceCascade)
        rects = face_detector(gray, frame)
        for rect in rects[0:1]:
                (startX, startY, endX, endY) = rect
                w = endX - startX
                h = endY - startY
                if w < 10 or h < 10:
                        continue
                #cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                euler_angle = pose.detection(gray, frame, rect)
                current_setPoint = euler_angle[0:3,0]
                clipped = np.clip(smoothed_angles[0:2]-face_setPoint,np.array([-15,-15]),np.array([15,15]))
                sm.send_msg(clipped[1]*6, -clipped[0]*6)           
                cv2.putText(frame, "X: " + "{:7.2f}".format(smoothed_angles[0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Y: " + "{:7.2f}".format(smoothed_angles[1]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Z: " + "{:7.2f}".format(smoothed_angles[2]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                
        smoothed_angles = (1-smoothing)*current_setPoint + smoothing*previous_angles
        previous_angles = smoothed_angles
        cv2.imshow('orig', frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop.
        if key == ord("q"):
                break
        elif key == ord("s"):
                face_setPoint = current_setPoint[0:2]
        
# Cleanup before exit.
cv2.destroyAllWindows()
vs.stop()
sm.stop_conn()

