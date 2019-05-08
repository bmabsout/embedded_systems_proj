import time

import cv2
import numpy as np
from imutils.video import VideoStream
import imutils

# Are we using the Pi Camera?
usingPiCamera = True
# Set initial frame size.
frameSize = (320, 240)

# Initialize mutithreading the video stream.
vs = VideoStream(src=0, usePiCamera=usingPiCamera, resolution=frameSize,
		framerate=24).start()
# Allow the camera to warm up.
time.sleep(2.0)
faceCascade = cv2.CascadeClassifier('face2.xml')
eyeCascade = cv2.CascadeClassifier('eye.xml')
timeCheck = time.time()
while True:
	# Get the next frame.
	frame = vs.read()
	
	# If using a webcam instead of the Pi Camera,
	# we take the extra step to change frame size.
	if not usingPiCamera:
		frame = imutils.resize(frame, width=frameSize[0])
	#frame = imutils.resize(frame, height=500, width=500)
	# Show video stream
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
			gray,
			scaleFactor=1.2,
			minNeighbors=5,
			minSize=(50, 50)
		)
	for (x,y,w,h) in faces[0:1]:
	# Create circle around faces
		cv2.circle(frame, (int((x + x + w)/2), int((y + y + h)/2)), int(h/2), (0, 255, 0), 5)
		gray_face = gray[y:y+h, x:x+w]
		scale = 1.5
		gray_face = imutils.resize(gray_face,int(w*scale),int(h*scale))
		eyes = eyeCascade.detectMultiScale(gray_face, 1.2, 5)[0:2]
		for (x2,y2,w2,h2) in eyes:
			# Create circle around eyes
			cv2.circle(frame, (int((x + x + (x2 + x2 +w2)/scale)/2), int((y + y + (y2 + y2 + h2)/scale)/2)), int(h2/(2*scale)), (255, 255, 0), 2)
		
	#eyes = eyeCascade.detectMultiScale(gray,1.2, 5)[0:2]
	#for (x,y,w,h) in eyes:
	# Create circle around eyes
	#	cv2.circle(frame, (int((x + x + w)/2), int((y + y + h)/2)), int(h/2), (255, 255, 0), 2)
	cv2.imshow('orig', frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop.
	if key == ord("q"):
		break
	
	#print(1/(time.time() - timeCheck))
	#timeCheck = time.time()

# Cleanup before exit.
cv2.destroyAllWindows()
vs.stop()
