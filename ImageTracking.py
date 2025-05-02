from collections import deque
from imutils.video import VideoStream
import numpy as np
import pandas as pd
import cv2
import imutils
import time

import numpy as np

# BLUE ping pong ball
color_upper = (99, 255, 255)
color_lower = (49, 92, 118)

pts = deque(maxlen=64)
vs = VideoStream(src=0).start()
vs2 = VideoStream(src=2).start()

N = 10
dataList = []
data = []
targets = []

step = 0
while True:
    frame = vs.read()
    frame2 = vs2.read()

    frame = imutils.resize(frame, width=600)
    frame2 = imutils.resize(frame2, width=600)

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    blurred2 = cv2.GaussianBlur(frame2, (11, 11), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(blurred2, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, color_lower, color_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    mask2 = cv2.inRange(hsv2, color_lower, color_upper)
    mask2 = cv2.erode(mask2, None, iterations=2)
    mask2 = cv2.dilate(mask2, None, iterations=2)
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    cnts2 = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)
    
    center = None
    center2 = None

    if len(cnts2) > 0:
        c2 = max(cnts2, key=cv2.contourArea)
        ((x2, y2), radius2) = cv2.minEnclosingCircle(c2)
        # print(x2, y2)

        if y2 > 170 and step == N:
            # if 120 < x2 < 445:
            landing_point = ((-x2 + 415) / (-415 + 116)) * -100
            print(landing_point)
            confirm = input('Confirm?')
            if confirm == "":
                dataList.append(data.copy())
                targets.append(int(round(landing_point)))

            step = 0
            data = []

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        M = cv2.moments(c)
        center = ((M["m10"] / M["m00"]), (M["m01"] / M["m00"]))
        
        if radius > 7.5 and step < N:
            data.append([center[0], center[1], radius])
            step += 1

        # if step == N:
        #     print(len(data))
            

        #     # target = input('Enter distance: ')
        #     # targets.append(target)


        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

    pts.appendleft(center)

    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue

        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    cv2.imshow("Frame", frame)
    cv2.imshow("Frame2", frame2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
vs2.stop()
cv2.destroyAllWindows()


df = pd.DataFrame({"data" : dataList, "target": targets})
print(df)
df.to_csv('new_dataset.csv', mode='a', header=False)