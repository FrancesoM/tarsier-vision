import cv2
import gi
import threading
import queue
import numpy as np
import time
from collections import deque

gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)

# ----------------------------
# Thread-safe queues for communication
# ----------------------------
trigger_queue = queue.Queue()
done_queue = queue.Queue()


# ----------------------------
# TPU inference placeholder
# ----------------------------
def tpu_inference(frame):
    """
    Replace this with actual TPU model inference.
    Returns True if a person is detected.
    """
    return np.random.rand() > 0.8  # simulation


# ----------------------------
# Save video utility
# ----------------------------
def save_video(frames, filename='event.mp4', fps=30):
    if not frames:
        return
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for f in frames:
        out.write(f)
    out.release()
    print(f"[INFO] Saved video {filename}")


# ----------------------------
# Low-res motion detection thread
# ----------------------------
def low_res_motion_thread():
    low_res_pipeline = (
        "rtspsrc location=rtsp://10.10.1.108:456/slow latency=100 ! "
        "rtpjitterbuffer ! "
        "rtph265depay ! "
        "h265parse ! v4l2h265dec ! "
        "videoconvert ! videoscale ! videorate ! "
        "video/x-raw,width=320,height=320,framerate=10/1 ! appsink drop=1"
    )

    cap = cv2.VideoCapture(low_res_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[ERROR] Failed to open low-res pipeline")
        return

    frame_queue = deque(maxlen=2)  # keep last 2 frames for motion detection

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        frame_queue.append(gray)

        # Only detect when we have 2 frames
        if len(frame_queue) < 2:
            continue

        prev_frame, curr_frame = frame_queue[0], frame_queue[1]
        frame_delta = cv2.absdiff(prev_frame, curr_frame)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        motion_area = cv2.countNonZero(thresh)

        if motion_area > 500:  # tune as needed
            person_detected = tpu_inference(frame)
            if person_detected:
                print("[INFO] Person detected, triggering event")
                trigger_queue.put(True)
                done_queue.get()  # wait for high-res thread to finish
                frame_queue.clear()  # reset queue to avoid false motion


# ----------------------------
# High-res capture thread
# ----------------------------
def high_res_capture_thread():
    high_res_pipeline = (
        "rtspsrc location=rtsp://10.10.1.108:456/highres latency=100 ! "
        "rtpjitterbuffer ! "
        "rtph265depay ! "
        "h265parse ! v4l2h265dec ! "
        "videoconvert ! appsink drop=1"
    )

    cap = cv2.VideoCapture(high_res_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("[ERROR] Failed to open high-res pipeline")
        return

    buffer_length = 10 * 30  # 10s buffer at 30fps (adjust if actual fps differs)
    frame_buffer = deque(maxlen=buffer_length)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame_buffer.append(frame)

        # Wait for a trigger from low-res thread
        if not trigger_queue.empty():
            trigger_queue.get()  # consume trigger

            print("[INFO] Capturing event: past buffer + 1 min")
            past_frames = list(frame_buffer)

            future_frames = []
            start_time = time.time()
            while time.time() - start_time < 60:  # 1 min future
                ret, f_frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                future_frames.append(f_frame)

            all_frames = past_frames + future_frames
            save_video(all_frames)

            done_queue.put(True)  # notify low-res thread to resume


# ----------------------------
# Main program
# ----------------------------
if __name__ == "__main__":
    t1 = threading.Thread(target=low_res_motion_thread, daemon=True)
    t2 = threading.Thread(target=high_res_capture_thread, daemon=True)

    t1.start()
    t2.start()

    # Keep main alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] Exiting program")
