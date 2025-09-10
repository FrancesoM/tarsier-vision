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

def high_res_capture_thread(pwd):

    print(f"Using password {pwd}")

    high_res_pipeline = (
        f"rtspsrc rtsp://admin:{pwd}@10.10.1.108:554/cam/realmonitor?channel=1&subtype=0latency=100 ! "
        "rtpjitterbuffer ! "
        "rtph265depay ! "
        "h265parse ! v4l2h265dec ! "
        "videoconvert ! appsink drop=1"
    )

    cap = cv2.VideoCapture(high_res_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[ERROR] Failed to open high-res pipeline")
        return

    buffer_length = 10 * 25  # 10s buffer at 30fps (adjust if actual fps differs)
    frame_buffer = deque(maxlen=buffer_length)


    # Capture 10s
    start_time = time.time()
    while time.time() - start_time < 10:  
        ret, frame = cap.read()
        frame_buffer.append(frame)

    save_video(frame_buffer)

high_res_capture_thread(sys.argv[1])



