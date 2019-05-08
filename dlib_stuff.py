import time

import cv2
import numpy as np
from imutils.video import VideoStream
from imutils import face_utils
import face_recognition as fr
import dlib
import imutils

# Set initial frame size.
frameSize = (320, 320)

# Initialize mutithreading the video stream.
vs = VideoStream(src=0, usePiCamera=True, resolution=frameSize,
                framerate=24).start()

faceCascade = cv2.CascadeClassifier('face4.xml')
#eyeCascade = cv2.CascadeClassifier('eye.xml')
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
detector = dlib.get_frontal_face_detector()


K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]


def get_head_pose(shape):
    gray_face = gray[startY:endY, startX:endX]
    scale = 1
    w = int(scale*(endX - startX))
    h = int(scale*(endY - startY))
    gray_face = imutils.resize(gray_face,w,h)
    shape = predictor(gray_face, dlib.rectangle(0,0,w,h))
    shape = face_utils.shape_to_np(shape)
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle


def initDNN():
        modelFile = "opencv_face_detector_uint8.pb"
        configFile = "opencv_face_detector.pbtxt"
        return cv2.dnn.readNetFromTensorflow(modelFile, configFile)

net = initDNN()

def runCascade(gray, cascade):
        faces = cascade.detectMultiScale(gray,1.3,5,minSize=(50,50))
        def rectConverter(t):
                (x,y,w,h) = t
                return (x,y,x+w,y+h)
        return map(rectConverter,faces)

def runDNN(image,net, confidence_thresh=0.15):
        (h, w) = image.shape[:2]
        new_size = (60,60)
        blob = cv2.dnn.blobFromImage(cv2.resize(image, new_size), 1.0,
        new_size, (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        rects = []
        for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections[0, 0, i, 2]
                # filter out weak detections by ensuring the `confidence` is
                # greater than the minimum confidence
                if confidence > confidence_thresh:
                        # compute the (x, y)-coordinates of the bounding box for the
                        # object
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        if endX - startX < w and endY - startY < h:
                                rects.append(box.astype("int"))
                                # draw the bounding box of the face along with the associated
                                # probability
                                text = "{:.2f}%".format(confidence * 100)
                                y = startY - 10 if startY - 10 > 10 else startY + 10
                                cv2.putText(image, text, (startX, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        return rects

def runDlibDetector(gray):
        faces = detector(gray,0)
        def rectConverter(t):
                (i,(x,y,w,h)) = t
                return (rect.left(),rect.bottom(),rect.right(),rect.top())
        return map(rectConverter,enumerate(faces))
        

def runLandmarks(gray):
        gray_face = gray[startY:endY, startX:endX]
        scale = 1
        w = int(scale*(endX - startX))
        h = int(scale*(endY - startY))
        gray_face = imutils.resize(gray_face,w,h)
        shape = predictor(gray_face, dlib.rectangle(0,0,w,h))
        shape = face_utils.shape_to_np(shape)
        for (x,y) in shape:
                cv2.circle(frame, (int(startX + x/scale),int(startY + y/scale)), 2, (0,255,0), -1)
angle_list = []
while True:
        # Get the next frame.
        frame = vs.read()
        frame = cv2.flip(frame, 1)
        #frame = imutils.resize(frame, height=500, width=500)
        # Show video stream
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #rects = runCascade(gray, faceCascade)
        rects = runDNN(frame, net)
        for (startX, startY, endX, endY) in rects:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                runLandmarks(gray)
                reprojectdst, euler_angle = get_head_pose(gray)
                angle_list.append(euler_angle)
                for start, end in line_pairs:
                    cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

                cv2.putText(frame, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
                cv2.putText(frame, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 0, 0), thickness=2)
##      rects = detector(gray,0)
##      for (i, rect) in enumerate(rects):
##              cv2.circle(frame, (rect.center().x, rect.center().y), int(rect.height()/2), (255,0,0),3)
##              shape = predictor(gray, rect)
##              shape = face_utils.shape_to_np(shape)
##              for (x,y) in shape:
##                      cv2.circle(frame, (x,y), 2, (0,255,0), -1)
        cv2.imshow('orig', frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop.
        if key == ord("q"):
                break
        
# Cleanup before exit.
cv2.destroyAllWindows()
vs.stop()
with open('text_file.txt', 'w') as f:
    for item in angle_list:
        f.write("%s\n" % item)
