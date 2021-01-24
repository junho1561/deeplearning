import numpy as np
import pandas as pd
import imutils
import time
import timeit
import dlib
import cv2
import math
from face_detector import get_face_detector, find_faces
from face_landmarks import get_landmark_model, detect_marks
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from threading import Timer
from check_cam_fps import check_fps
import make_train_data as mtd
import light_remover as lr
import ringing_alarm as alarm
from sklearn.preprocessing import MinMaxScaler


def con():
############################################################################################
            # 눈
############################################################################################
    def eye_aspect_ratio(eye) :
        eA = dist.euclidean(eye[1], eye[5]) 
        eB = dist.euclidean(eye[2], eye[4]) 
        eC = dist.euclidean(eye[0], eye[3])
        ear = (eA + eB) / (2.0 * eC) 
        return ear

    def init_open_ear() : #눈을 떴을 때
        time.sleep(5)
        print('뜬눈을 측정합니다')
        ear_list = []
        for i in range(7) :
            ear_list.append(both_ear) 
            time.sleep(1)
        global OPEN_EAR 
        OPEN_EAR = sum(ear_list) / len(ear_list) 
        
    def init_close_ear() :
        time.sleep(2)
        print('close eye')
        th_opene.join() 
        time.sleep(5)
        ear_list = []
        time.sleep(1)
        for i in range(7) :
            ear_list.append(both_ear)
            time.sleep(1)
        CLOSE_EAR = sum(ear_list) / len(ear_list)
        global EAR_THRESH
        EAR_THRESH = (((OPEN_EAR - CLOSE_EAR) / 2) + CLOSE_EAR) 
        
###########################################################################################################################################
            # 입
 ##########################################################################################################################################
        
    def yawn_ratio(mouth) : # 하품 계산하는 공식
        mA = dist.euclidean(mouth[13], mouth[19]) # 세로 길이
        mB = dist.euclidean(mouth[15], mouth[17]) # 세로 길이
        mC = dist.euclidean(mouth[0], mouth[6]) # 가로 길이
        mr = (mA + mB) / (2.0 * mC)
        return mr 
############################################################################################################################################
      # 고개돌리기
############################################################################################################################################
    def get_2d_points(frame, rotation_vector, translation_vector, camera_matrix, val):
        """Return the 3D points present as 2D for making annotation box"""
        point_3d = []
        dist_coeffs = np.zeros((4,1))
        rear_size = val[0]
        rear_depth = val[1]
        point_3d.append((-rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, rear_size, rear_depth))
        point_3d.append((rear_size, -rear_size, rear_depth))
        point_3d.append((-rear_size, -rear_size, rear_depth))

        front_size = val[2]
        front_depth = val[3]
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d.append((-front_size, front_size, front_depth))
        point_3d.append((front_size, front_size, front_depth))
        point_3d.append((front_size, -front_size, front_depth))
        point_3d.append((-front_size, -front_size, front_depth))
        point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

        # Map to 2d img points
        (point_2d, _) = cv2.projectPoints(point_3d,
                                          rotation_vector,
                                          translation_vector,
                                          camera_matrix,
                                          dist_coeffs)
        point_2d = np.int32(point_2d.reshape(-1, 2))
        return point_2d

    
    def head_pose_points(frame, rotation_vector, translation_vector, camera_matrix):
        """
        Get the points to estimate head pose sideways    

        Parameters
        ----------
        frame : np.unit8
            Original Image.
        rotation_vector : Array of float64
            Rotation Vector obtained from cv2.solvePnP
        translation_vector : Array of float64
            Translation Vector obtained from cv2.solvePnP
        camera_matrix : Array of float64
            The camera matrix

        Returns
        -------
        (x, y) : tuple
            Coordinates of line to estimate head pose

        """
        rear_size = 1
        rear_depth = 0
        front_size = frame.shape[1]
        front_depth = front_size*2
        val = [rear_size, rear_depth, front_size, front_depth]
        point_2d = get_2d_points(frame, rotation_vector, translation_vector, camera_matrix, val)
        y = (point_2d[5] + point_2d[8])//2
        x = point_2d[2]

        return (x, y)


############################################################################################
            # 변수 코드
############################################################################################
    OPEN_EAR = 0 
    EAR_THRESH = 0 


    EAR_CONSEC_FRAMES = 25  
    COUNTER = 0 
    
    closed_eyes_time = [] 
    TIMER_FLAG = False 
    ALARM_FLAG = False

    ALARM_COUNT = 0 
    RUNNING_TIME = 0 
    
    PREV_TERM = 0 

    np.random.seed(30)
    power, nomal, short = mtd.start(25) 
    test_data = []
    result_data = []
    prev_time = 0

    detector = dlib.get_frontal_face_detector() 
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
    
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


 ############################################################################################
            # 스레드
 ############################################################################################
    
    th_opene = Thread(target = init_open_ear)
    th_opene.deamon = True 
    th_opene.start() 
    th_closee = Thread(target = init_close_ear)
    th_closee.deamon = True 
    th_closee.start() 
#     th_openm = Thread(target = init_open_mouth)
#     th_openm.deamon = True 
#     th_openm.start() 
#     th_closem = Thread(target = init_close_mouth)
#     th_closem.deamon = True 
#     th_closem.start() 

    global both_ear_list
    both_ear_list = []
    test = []
    both_ear=0
    
    minMax = MinMaxScaler()
    mr_list = []
    mr_scaled_list = []
    
    head_angle_list = []
    
    global Concentration
    Concentration = []
    
    global count
    count = 0
    
    global ctime
    ctime_list = []
    global  ctime_result_list 
    ctime_result_list=[]
    vtime = []
    
    ret, frame = cap.read()
    face_model = get_face_detector()
    landmark_model = get_landmark_model()
    size = frame.shape
    pre_time = 0
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, -65.0),        # Chin
                                (-225.0, 170.0, -135.0),     # Left eye left corner
                                (225.0, 170.0, -135.0),      # Right eye right corne
                                (-150.0, -150.0, -125.0),    # Left Mouth corner
                                (150.0, -150.0, -125.0)      # Right mouth corner
                            ])

    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )
    
    while True:
        ctime = timeit.default_timer()
        if len(vtime) == 0:
            vtime.append(ctime)
        ret, frame = cap.read()
        
        ## 시간반복을 주기 위해
        if pre_time == 0 :
            pre_time = time.time()
            
        current_time = time.time()
        
        if not ret:
            print('비디오 읽기 오류')
            break
        
        frame = cv2.flip(frame,1)
        frame = imutils.resize(frame, width = 400) # 이미지 크기재조정
        L, gray = lr.light_removing(frame) # 조명 영향 최소화
        
        rects = detector(gray,0)
        faces = find_faces(frame, face_model)
        for face in faces:
            marks = detect_marks(frame, landmark_model, face)
            # mark_detector.draw_marks(frame, marks, color=(0, 255, 0))
            image_points = np.array([
                                    marks[30],     # Nose tip
                                    marks[8],     # Chin
                                    marks[36],     # Left eye left corner
                                    marks[45],     # Right eye right corne
                                    marks[48],     # Left Mouth corner
                                    marks[54]      # Right mouth corner
                                ], dtype="double")
            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
            
            
            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose
            
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            
            #for p in image_points:
                #cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
            
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            x1, x2 = head_pose_points(frame, rotation_vector, translation_vector, camera_matrix)

            #cv2.line(frame, p1, p2, (0, 255, 255), 2)
            #cv2.line(frame, tuple(x1), tuple(x2), (255, 255, 0), 2)
            # for (x, y) in marks:
            #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
            # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
            try:
                m = (p2[1] - p1[1])/(p2[0] - p1[0])
                ang = int(math.degrees(math.atan(m)))
                head_angle_list.append(ang)
            except:
                ang = 90

        for rect in rects:
            shape = predictor(gray, rect) #얼굴에서 랜드마크
            shape = face_utils.shape_to_np(shape) # 68개의 점 좌표

############################################################################################
            # 눈 
 ############################################################################################
            leftEye = shape[lStart:lEnd] #왼쪽 눈 점 좌표
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye) #왼쪽 눈 평균거리
            rightEAR = eye_aspect_ratio(rightEye)
            
            both_ear = (leftEAR + rightEAR) / 2  
            
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0,255,0), 1) 
            cv2.drawContours(frame, [rightEyeHull], -1, (0,255,0), 1)
                
            mouth = shape[mStart:mEnd] # 입
            mr = round(yawn_ratio(mouth), 2)   

            cv2.putText(frame, "EAR : {:.2f}".format(both_ear), (280,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)
            both_ear_list.append(both_ear)
        
            global eyes
            
            if(len(both_ear_list) >= 10):
                both_ear_pd = pd.DataFrame(both_ear_list)
                minMax.fit(both_ear_pd)
                
                eyes = minMax.transform(both_ear_pd)
                cv2.putText(frame, "Eye : {:.2f}".format(eyes[-1][0]), (280,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

            
############################################################################################
            # 입
 ############################################################################################


            mouthHull = cv2.convexHull(mouth)
            cv2.drawContours(frame, [mouthHull], -1, (0,255,0), 1)

            mr_list.append(mr)
            
            if len(mr_list)>=10:
                df_mr_list = pd.DataFrame(mr_list)
                minMax.fit(df_mr_list)
                global df_mr_scaled
                df_mr_scaled = minMax.transform(df_mr_list)
                cv2.putText(frame, "Mouth: {:.2f}".format(1-(df_mr_scaled[-1][0])), (280,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)

            
            

            cv2.putText(frame, "MAR : {:.2f}".format(mr), (280,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,30,20), 2)
            
            if(current_time - pre_time > 5.0) :
                if((len(both_ear_list) >= 10)):
                    Concentration.append( ((0.6) * eyes[-1][0]) + ((0.2) * (1-(df_mr_scaled[-1][0]))) + ((0.2)*(1-math.cos(head_angle_list[-1]))) )
                    
                pre_time = 0
                ctime_list.append(ctime)
                ctime_result_list.append(ctime_list[-1]-vtime[0])
#                 Concentration.append( ((0.7) * prob[-1][0]) + ((0.25) * (1-(df_mr_scaled[-1][0]))) + ((0.05) * pose[-1][0]) )  # 1/3는 가중치
#                 print("집중도 : ", Concentration[-1])
#                 print("눈 : ", prob[-1])
#                 print("입 : ", (1-(df_mr_scaled[-1])) )
#                 print()
#                 print("\n")
            
            
#######################################################################################################
        cv2.imshow("frame",frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break