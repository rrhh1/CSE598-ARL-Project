from collections import deque
from imutils.video import VideoStream
import numpy as np
import pandas as pd
import cv2
import imutils
import time

import numpy as np

# BLUE ping pong ball
color_upper = (255, 255, 255)
color_lower = (85, 42, 203)

pts = deque(maxlen=64)
vs = VideoStream(src=1).start()

N = 10
dataList = []
data = []
targets = []

step = 0
while True:
    frame = vs.read()

    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, color_lower, color_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        data.append([center[0], center[1], radius])
        step += 1

        if step == N:
            print(data)
            dataList.append(data.copy())
            data = []
            step = 0

            target = input('Enter distance: ')
            targets.append(target)


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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()

df = pd.DataFrame({"data" : dataList, "target": targets})
print(df)
df.to_csv('dataset.csv', mode='a', header=False)