import cv2
import numpy as np
import os
import torch
from torchvision.utils import draw_bounding_boxes

RTSP_URL = """YOUR_RTSP_URL"""

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    print('Cannot open RTSP stream')
    exit(-1)

frames = []
model = torch.load("""PATH_TO_YOUR_MODEL""")
count = 0
classes = ['street-trade', 'street-trade']
frames_path = """PATH_TO_FRAMES_DIR"""

while True:
    _, frame = cap.read()
    frame_path = frames_path
    prediction = model([frame])
    pred = prediction[0]
    frame = draw_bounding_boxes(frame,
                        pred['boxes'][pred['scores'] > 0.3],
                        [classes[i] for i in pred['labels'][pred['scores'] > 0.3].tolist()], width=4
                        ).permute(1, 2, 0)
    cv2.imwrite(frame_path, frame)
    frames.append(frame)
    count += 1
    if len(frames) > 300:
        break
