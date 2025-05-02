from collections import deque
from imutils.video import VideoStream
from joblib import load

from model import RNN_Model, Modified_ResNet
import numpy as np
import pandas as pd
import cv2
import imutils
import time
import serial

import torch
import torch.nn as nn

def run_prediction(model, data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = torch.tensor(data, dtype=torch.float32).to(device)
    data = nn.functional.normalize(data)
    data = data.view(1, 10, 3)

    return round(model(data).item() * 100)

def run_CNN_prediction(model, data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = torch.tensor(data, dtype=torch.float32).to(device)
    data = data.unsqueeze(0)

    outputs = model(data)
    predicted = torch.softmax(outputs, dim=1)
    predicted = torch.argmax(predicted, dim=1)

    return int(predicted.item())

def send_instruction(instruction, ser):
    
    if instruction < 0:
        instruction = 0
    
    if instruction > 100:
        instruction = 100

    ser.write(bytes([instruction]))

    response = ser.read(1)
    print(instruction)

    time.sleep(3)

    ser.write(b'\x32')


# BLUE ping pong ball
color_upper = (99, 255, 255)
color_lower = (49, 92, 118)

pts = deque(maxlen=64)
vs = VideoStream(src=0).start()

model = torch.load("Modified_ResNet.pth", weights_only=False)

N = 10
data = []

serial_port = serial.Serial("COM3", 115200)
time.sleep(2)
serial_port.write(b'\x32')

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
        
        if radius > 7.5:
            data.append([center[0], center[1], radius])
            step += 1

        if step == N:
            prediction = run_CNN_prediction(model, data)
            send_instruction(prediction, serial_port)
            
            data = []
            step = 0

            input('Pause')


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

serial_port.write(b'\x00')
time.sleep(2)
serial_port.close()