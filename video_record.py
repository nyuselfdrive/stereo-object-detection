'''
Record unrectified stereo video
'''

import cv2
import numpy as np

cap = cv2.VideoCapture(1)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

recording = False
out = cv2.VideoWriter("out.avi", fourcc, 20.0, (1344, 376))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        cv2.imshow("Recorder", frame)
        print(frame.shape)
        if recording:
            out.write(frame)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('s'):
            recording = not recording
            print("Now Recording" if recording else "Stopping recording")
        if key & 0xFF == ord('q'):
            print("goodbye")
            break

out.release()
cap.release()
        

        

   
