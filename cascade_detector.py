import cv2

def cascader(cascade_file):

    cascade = cv2.CascadeClassifier(cascade_file)
    def runCascade(gray, colored):
        faces = cascade.detectMultiScale(gray,1.3,5,minSize=(50,50))
        def rectConverter(t):
                (x,y,w,h) = t
                return (x,y,x+w,y+h)
        return list(map(rectConverter,faces))
    return runCascade


face_detector = cascader('face4.xml')
