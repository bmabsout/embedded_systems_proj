import imutils
from imutils import face_utils
import cv2
import numpy as np
import dlib



def run_landmarks(gray, rect,predictor):
    scale = 1    
    (startX, startY, endX, endY) = rect
    #cropped = gray[startY:endY, startX:endX]
    #w = int(scale*(endX - startX))
    #h = int(scale*(endY - startY))
    #gray_face = imutils.resize(cropped,w,h)
    shape = predictor(gray, dlib.rectangle(startX,startY,endX,endY))
    landmarks = face_utils.shape_to_np(shape)
    def unscale(xy):
        (x,y) = xy
        return (startX + x/scale, startY + y/scale)
    return landmarks

def draw_landmarks(landmarks, frame):
    for (x,y) in landmarks:
        cv2.circle(frame, (int(x),int(y)), 2, (0,255,0), -1)



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

def draw_cube(reprojectdst, frame):
    for start, end in line_pairs:
        (startX,startY) = reprojectdst[start]
        (endX, endY) = reprojectdst[end]
        if(abs(startX) > 10000 or abs(startY) > 10000 or abs(endX) > 10000 or abs(endY) > 10000):
           break
        cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))


def run_pose_detection(landmarks_dat):
    predictor = dlib.shape_predictor(landmarks_dat)
    
    def pose_detection_runner(gray, frame, rect, draw_l = True, draw_c = True):
        landmarks = run_landmarks(gray, rect,predictor)
        if draw_l:
            draw_landmarks(landmarks,frame)
            
        reprojectdst, euler_angle = get_head_pose(landmarks)
        
        if draw_c:
            draw_cube(reprojectdst,frame)
            
        return euler_angle
    
    return pose_detection_runner

detection = run_pose_detection('shape_predictor_68_face_landmarks.dat')
