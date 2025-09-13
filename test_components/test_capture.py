import cv2
import gi
import threading
import queue
import numpy as np
import time
from collections import deque
import sys

gi.require_version('Gst', '1.0')
from gi.repository import Gst

def save_video(frames, filename='event.mp4', fps=25):
    if not frames:
        return
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for f in frames:
        out.write(f)
    out.release()
    print(f"[INFO] Saved video {filename}")

def capture_thread(pwd,low_res=0):

    print(f"Using password {pwd}")
    
    if low_res:
        pipeline = (
            f"rtspsrc location=rtsp://admin:{pwd}@10.10.1.108:554/cam/realmonitor?channel=1&subtype=1 latency=100 protocols=tcp !"
            "rtpjitterbuffer ! "
            "rtph265depay ! "
            "h265parse ! avdec_h265 ! "
            "videoconvert ! videoscale ! videorate ! "
            "video/x-raw,width=320,height=320,framerate=10/1 ! appsink drop=1"
        )

        fps = 10 # TODO: get this as variable from the framerate downsampling 
    else:
        pipeline = (
            f"rtspsrc location=rtsp://admin:{pwd}@10.10.1.108:554/cam/realmonitor?channel=1&subtype=0 latency=100 protocols=tcp !"
            "rtpjitterbuffer ! "
            "rtph265depay ! "
            "h265parse ! avdec_h265 ! "
            "videoconvert ! appsink drop=1"
        )
        fps = 25 # TODO: is there a way to get this info from the stream itself?

    print(pipeline)

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[ERROR] Failed to open pipeline")
        return

    buffer_length = 10 * fps  # 10s buffer at selected fps (adjust if actual fps differs)
    frame_buffer = deque(maxlen=buffer_length)


    # Capture 10s
    start_time = time.time()
    while time.time() - start_time < 10:  
        ret, frame = cap.read()
        frame_buffer.append(frame)

    save_video(frame_buffer)

# Usage python test_capture.py <password> <1/0> 
capture_thread(pwd=sys.argv[1],low_res=sys.argv[2])



