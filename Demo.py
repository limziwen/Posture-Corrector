import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

frames = [] #array of frames

print("Enter sensitivity lower number higher sensitivity, lowest = 1")
num_frames_per_analysis = int(input())

print("Please get into a neutral posture, and press escape once you've found it")

for x in range (0, 2): 
  #first run for callibration
  
  
  count = 1
  with mp_pose.Pose(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5) as pose:

      while cap.isOpened():
        
        success, image = cap.read()

        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = pose.process(image)

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('MediaPipe Pose', image)

        #Press escape to leave
        if cv2.waitKey(5) & 0xFF == 27:
          callibrated_frame = results.pose_landmarks.landmark
          break

        if x == 1:
          count = count + 1
          if count % num_frames_per_analysis == 0:

            #Left, Right Shoulder
              # can check the y value 

            #Head positions (can use heuristic like the nose)

            # print(results.pose_landmarks.landmark[11].z) #left shoulder
            # print(results.pose_landmarks.landmark[12].z) #right shoulder

            left_shoulder = results.pose_landmarks.landmark[11]
            right_shoulder = results.pose_landmarks.landmark[12]

            avg_shoulder_z = (left_shoulder.z + right_shoulder.z)/2
            callibrated_avg_shoulder_z = (callibrated_frame[11].z + callibrated_frame[12].z)/2
            
            # print(" shoulder z diff: ", callibrated_avg_shoulder_z - avg_shoulder_z)

            if(abs(callibrated_avg_shoulder_z - avg_shoulder_z) > 0.2): #0.2 just ad hoc
              print("Please re-adjust your posture! Shoulders are moving too far from original postion")

            nose = results.pose_landmarks.landmark[0]

            # print("nose displacement ", abs(nose.y - callibrated_frame[0].y))
            if (callibrated_frame[0].y - nose.y) > 0.1 :
              print("Please re-adjust your posture! Lift up your head to your original postion")

            


cap.release()
cv2.destroyAllWindows()

#append all landmark data
# frames.append(results.pose_landmarks.landmark) 

# if count % num_frames_per_analysis ==0:
#   print(results.pose_landmarks.landmark[11].z) #left shoulder
#   print(results.pose_landmarks.landmark[12].z) #right shoulder
#   cv2.waitKey(-1)
#   # print("average is: ", np.average(frames))

#Check criteria 1: z distance from left and right shoulder 
  #Heuristic 1: Take the 
#Check criteria 2: y distance from head components
# frames.clear()