import dlib

def dliber():
    detector = dlib.get_frontal_face_detector()
    def runDlibDetector(gray,image):
            faces = detector(gray,0)
            print(faces);
            def rectConverter(convert_me):
                    (i,rect) = convert_me
                    return (rect.left(),rect.bottom(),rect.right(),rect.top())
            return map(rectConverter,enumerate(faces))
    return runDlibDetector
        
face_detector = dliber()
