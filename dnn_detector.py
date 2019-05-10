import cv2
import numpy as np

def dnner(modelFile, configFile):
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    def runDnn(gray,image,draw=False, confidence_thresh=0.15):
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
                                if draw:
                                    text = "{:.2f}%".format(confidence * 100)
                                    y = startY - 10 if startY - 10 > 10 else startY + 10
                                    cv2.putText(image, text, (startX, y),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        return rects
    
    return runDnn


modelFile = "opencv_face_detector_uint8.pb"
configFile = "opencv_face_detector.pbtxt"
face_detector = dnner(modelFile, configFile)
