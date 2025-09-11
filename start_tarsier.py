from datetime import datetime
import sys
import os
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

import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(threadName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# TPU imports
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
#from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import argparse


# ----------------------------
# Thread-safe queues for communication
# ----------------------------
trigger_queue = queue.Queue()
done_queue = queue.Queue()

# ----------------------------
# Save video utility
# ----------------------------
def save_encoded_video(encoded_buffers, filename):
    """
    Saves a list of H.265 encoded frames to a MKV file using GStreamer.
    """
    tmp_file = "tmp.h265"
    
    # Write all encoded buffers to a temporary raw file
    with open(tmp_file, "wb") as f:
        for buf in encoded_buffers:
            f.write(buf)
    
    # Use gst-launch to mux into MKV
    os.system(f'gst-launch-1.0 filesrc location={tmp_file} ! h265parse ! matroskamux ! filesink location={filename}')
    
    # Remove temporary raw file
    os.remove(tmp_file)



def load_labels(path):
    labels = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            labels.append(l.strip("\n"))
    return labels

# ----------------------------
# Low-res motion detection thread
# ----------------------------
def low_res_motion_thread():

    pwd = "gregoriosonia1!"

    # TPU setup 
    model_cpu ="/models/efficientdet_lite0_320_ptq.tflite"
    model_tpu ="/models/efficientdet_lite0_320_ptq_edgetpu.tflite"
    labels    ="/models/coco_labels.txt"

    logging.info("TPU: Setting the delegate")
    delegate = load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')

    # Load the model
    logging.info("TPU: Loading the model")
    interpreter = Interpreter(model_path=model_tpu, experimental_delegates=[delegate])
    interpreter.allocate_tensors()

    # Find class for "person label"
    labels_lookup = load_labels(labels)
    person_idx = 0
    for i in range(len(labels_lookup)):
        if labels_lookup[i] == "person":
            person_idx = i

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Pipeline setup
    low_res_pipeline = (
        f"rtspsrc location=rtsp://admin:{pwd}@10.10.1.108:554/cam/realmonitor?channel=1&subtype=1 latency=100 protocols=tcp !"
        "rtpjitterbuffer ! "
        "rtph265depay ! "
        "h265parse ! avdec_h265 ! "
        "videoconvert ! videoscale ! videorate ! "
        "video/x-raw,width=320,height=320,framerate=10/1 ! appsink drop=1"
    )

    logging.info(f"Starting pipeline as {low_res_pipeline}")

    cap = cv2.VideoCapture(low_res_pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        logging.info("[ERROR] Failed to open low-res pipeline")
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

        if motion_area > 200:  # tune as needed

            logging.info(f"Detected motion! Starting inference.. ")

            # frame is already the right dimension, let's just convert to 8bits
            input_data = np.expand_dims(np.array(frame, dtype=np.uint8), axis=0)

            # perform inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # parse results
            boxes = interpreter.get_tensor(output_details[0]['index'])[0]    # (num, 4)
            classes = interpreter.get_tensor(output_details[1]['index'])[0]  # (num,)
            scores = interpreter.get_tensor(output_details[2]['index'])[0]   # (num,)
            num = int(interpreter.get_tensor(output_details[3]['index'])[0])

            person_score = scores[person_idx]

            logging.info(f"Detected {num} objects")
            logging.info(f"Person was in frame with confidence {person_score}")

            if person_score > 0.25:
                logging.info("Person detected! Saving video buffer.. ")
                trigger_queue.put(True)
                done_queue.get()  # wait for high-res thread to finish
                frame_queue.clear()  # reset queue to avoid false motion
                logging.info(f"Restarting motion detection..")


# ----------------------------
# High-res capture thread
# ----------------------------
def high_res_capture_thread():
    pwd = "gregoriosonia1!"
    high_res_pipeline = (
        f"rtspsrc location=rtsp://admin:{pwd}@10.10.1.108:554/cam/realmonitor?channel=1&subtype=0 latency=100 protocols=tcp !"
        "rtpjitterbuffer ! "
        "rtph265depay ! "
        "h265parse ! appsink name=sink drop=1"
    )

    logging.info(f"Starting pipeline as {high_res_pipeline}")

    pipeline = Gst.parse_launch(high_res_pipeline)
    appsink = pipeline.get_by_name("sink")
    appsink.set_property("emit-signals", True)
    appsink.set_property("sync", False) 

    pipeline.set_state(Gst.State.PLAYING)

    buffer_length = 5 * 20  # 10s buffer at 25fps (adjust if actual fps differs)
    frame_buffer = deque(maxlen=buffer_length)

    def pull_encoded_buffer():
        """
        Pull an encoded frame from appsink.
        Returns bytes or None if no sample is available.
        """
        sample = appsink.emit("try-pull-sample", Gst.SECOND // 5)  # timeout ~200ms
        if sample:
            buf = sample.get_buffer()
            return buf.extract_dup(0, buf.get_size())
        return None


    while True:
        data = pull_encoded_buffer()
        if data:
            frame_buffer.append(data)

        # Wait for a trigger from low-res thread
        if not trigger_queue.empty():
            trigger_queue.get()  # consume trigger

            logging.info(f"Capturing event: past buffer + 1 min ")

            past_frames = list(frame_buffer)

            future_frames = []
            start_time = time.time()
            while time.time() - start_time < 10:  # 1 min future
                f_data = pull_encoded_buffer()
                if f_data:
                    future_frames.append(f_data)

            all_frames = past_frames + future_frames
            filename=datetime.now().strftime("event_%Y-%m-%d_%H-%M-%S") + ".mkv"

            save_encoded_video(all_frames,filename)

            logging.info(f"Video saved, restarting detection thread.. ")

            done_queue.put(True)  # notify low-res thread to resume


# ----------------------------
# Main program
# ----------------------------
if __name__ == "__main__":
    t1 = threading.Thread(target=low_res_motion_thread  , daemon=True)
    t2 = threading.Thread(target=high_res_capture_thread, daemon=True)

    # Set thread names when creating threads
    t1.name = "LowResThread"
    t2.name = "HighResThread"

    t1.start()
    t2.start()

    # Keep main alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[INFO] Exiting program")
