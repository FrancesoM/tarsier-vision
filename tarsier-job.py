#!/usr/bin/env python3
"""
High-res rolling recorder in RAM + low-res TPU motion detection combined.

Behavior:
- Starts a gst-launch pipeline that writes split MKV segments to /dev/shm/highres_segments/
  (each segment_time_secs long).
- Keeps only the latest `max_segments` in the RAM dir (cleanup thread).
- Runs a low-res OpenCV pipeline (320x320 @10fps) for motion detection + TPU inference.
- On person detection: wait post_buffer_secs (60s), then copy the latest `max_segments`
  from RAM to a timestamped folder on disk (events_dir). The copy job is handed to a
  separate "stitch/send" worker thread via an event_queue.
"""

import os
import pathlib
from pathlib import Path
import sys
import time
import logging
import shutil
import threading
import subprocess
from datetime import datetime
from collections import deque
from queue import Queue
from urllib.parse import quote
import requests
from threading import Lock

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)


import cv2
import numpy as np

# TPU imports
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
from PIL import Image, ImageDraw, ImageFont

# ------------- Configuration -------------
RTSP_USER = "admin"
RTSP_PWD  = os.getenv('CAM_PWD')
RTSP_HOST = os.getenv('CAM_IP')
RTSP_PORT = 554
RTSP_PATH = "/cam/realmonitor?channel=1&subtype={}"

# Directories
RAM_DIR    = Path("/dev/shm/highres_segments")    # RAM-backed directory
EVENTS_DIR = Path("/workspace/events")           # where final copies are stored
RAM_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

# Remove old segments when this script is re-run
for f in RAM_DIR.iterdir():
    if f.is_file():
        f.unlink()

# Segment / buffering parameters
SEGMENT_TIME_SECS = 10                         # each MKV segment length
PRE_BUFFER_SECS = 120                          # total rolling buffer to keep (2 minutes)
POST_BUFFER_SECS = 60                          # after trigger, wait this much to copy future
MAX_SEGMENTS = max(1, PRE_BUFFER_SECS // SEGMENT_TIME_SECS)

# Low-res pipeline (for motion detection / TPU)
LOW_RES_RTSP = f"rtsp://{RTSP_USER}:{RTSP_PWD}@{RTSP_HOST}:{RTSP_PORT}{RTSP_PATH.format(1)}"
LOW_RES_PIPELINE = (
    f'rtspsrc location="{LOW_RES_RTSP}" latency=100 protocols=tcp ! '
    'rtpjitterbuffer ! rtph265depay ! h265parse ! avdec_h265 ! '
    'videoconvert ! videoscale ! videorate ! '
    'video/x-raw,width=320,height=320,framerate=10/1 ! appsink drop=1'
)

# High-res recording pipeline (split into MKV files in RAM directory)
HIGH_RES_RTSP = f"rtsp://{RTSP_USER}:{RTSP_PWD}@{RTSP_HOST}:{RTSP_PORT}{RTSP_PATH.format(0)}"
# We'll call gst-launch-1.0 with splitmuxsink location=RAM_DIR/segment_%05d.mkv max-size-time=... (ns)
# using protocols=tcp to be robust
HIGH_RES_PIPELINE = (
    f'rtspsrc location="{HIGH_RES_RTSP}" latency=100 protocols=tcp ! '
    f'rtpjitterbuffer ! rtph265depay ! h265parse ! '
    f'splitmuxsink location={RAM_DIR}/segment_%06d.mkv max-size-time={SEGMENT_TIME_SECS * 1_000_000_000}'
)

# Event queue for post-processing (stitch/send)
event_queue = Queue()

# Logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(threadName)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Global object for the TPU, shared with a mutex between threads. Each thread use the same model 
# TPU setup 
def load_labels(path):
    labels = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            labels.append(l.strip("\n"))
    return labels

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

# Create the mux 
tpu_lock = Lock()

# Each thread will then call this function for inferene
def run_tpu_inference(input_data):
    with tpu_lock:

        logging.debug(f"Acquired TPU lock, running inference!")
        # perform inference
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # parse results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]    # (num, 4)
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # (num,)
        scores = interpreter.get_tensor(output_details[2]['index'])[0]   # (num,)
        num = int(interpreter.get_tensor(output_details[3]['index'])[0])

        return boxes,classes,scores,num



def stitch_segments(segments_dir, output_dir):
    """
    Stitch MKV segments into a single output file using ffmpeg.
    segments_dir: folder containing segment files (e.g., /dev/shm/recorder/)
    output_dir: where to save final stitched file
    """
    # Collect and sort segments by name (splitmuxsink produces numbered parts)
    segments = sorted(
        [f for f in segments_dir.iterdir() if f.suffix == ".mkv"]
    )
    if not segments:
        logging.error("No MKV segments found to stitch.")
        return None

    # Create a temporary file list for ffmpeg
    filelist_path = segments_dir / "segments.txt"
    with open(filelist_path, "w") as f:
        for seg in segments:
            f.write(f"file '{seg.as_posix()}'\n")

    # Output file path
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = output_dir / f"event_{ts}.mp4"

    # Run ffmpeg (stream copy, with re-encoding)
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", filelist_path.as_posix(),
        "-c:v", "libx264",      # re-encode video to H.264
        "-c:a", "aac",          # re-encode audio to AAC
        "-preset", "fast",      # optional, speeds up encoding
        "-movflags", "+faststart",  # ensures MP4 is streamable
        output_file.as_posix()             # final output file
    ]

    logging.info(f"Stitching {len(segments)} segments into {output_file} with command {cmd}")
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logging.error(f"ffmpeg failed: {e.stderr.decode()}")
        return None

    # Cleanup of the segments
    for seg in segments:
        seg.unlink()

    logging.info(f"Final stitched file ready: {output_file}")
    return output_file


def copy_segments_to_event(ram_dir, events_dir, timestamp_str):
    """Copy latest max_segments from ram_dir into a new event folder and return path."""
    
    event_folder = events_dir / f"event_{timestamp_str}"
    event_folder.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ram_dir, event_folder, dirs_exist_ok=True)

    return event_folder

TG_TOKEN = os.getenv("TG_TOKEN")  # token from env
CHAT_ID = os.getenv("CHAT_ID", "").split(",")  # allow multiple IDs, comma separated

def send_text(text):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    
    r = requests.post(url, data={"chat_id": CHAT_ID,"text": text})
    logging.info(f"Sent to {CHAT_ID}: {r.status_code} {r.text}")

def send_photo(image_path, caption=""):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto"
    with open(image_path, "rb") as photo:
        r = requests.post(url, data={"chat_id": CHAT_ID, "caption": caption}, files={"photo": photo})
        logging.info(f"Sent frame to {CHAT_ID}: {r.status_code}")

def send_video(video_path):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendVideo"
    with open(video_path, "rb") as video:
        r = requests.post(url, data={"chat_id": CHAT_ID}, files={"video": video})
        logging.info(f"Sent to {CHAT_ID}: {r.status_code} {r.text}")

def check_for_person(video_path):
    cap = cv2.VideoCapture(video_path.as_posix())
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps // 5) if fps > 5 else 1  # sample ~5fps

    frame_idx = 0
    detected_frame_path = None
    person_timestamp    = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Convert OpenCV frame (BGR) -> PIL (RGB)
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Resize for TPU input
            in_shape = input_details[0]['shape']
            img_resized = img.resize((in_shape[2], in_shape[1]))
            input_data = np.expand_dims(np.array(img_resized, dtype=np.uint8), axis=0)
        
            boxes,classes,scores,num = run_tpu_inference(input_data)

            ############ DEBUG: Log what was found, for debugging ##############################################
            #classes_and_scores = [{"class":c , "score":s } for c,s in zip(classes,scores)]

            #classes_and_scores.sort(key=lambda x:x["score"]) # Sort by scores and print highest 10

            #for item in classes_and_scores[-5:]:
            #    logging.debug(f"{item["class"]}:{labels_lookup[int(item["class"])]} -> {item["score"]:.2f}")
            ####################################################################################################

            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=32)

            person_found        = False
            person_object_index = None 
            for i in range(num):
                if (int(classes[i]) == person_idx) and (scores[i] > 0.25):
                    person_found        = True
                    person_object_index = i

                    logging.info(f"Person found at frame {frame_idx} with score {scores[i]}")

            if person_found:
                ymin, xmin, ymax, xmax = boxes[person_object_index]
                (left, right, top, bottom) = (
                    xmin * img.width,
                    xmax * img.width,
                    ymin * img.height,
                    ymax * img.height,
                )

                text_label = f"person {scores[person_object_index]:.2f}"
                draw.rectangle([left, top, right, bottom], outline="red", width=3)
                draw.text((left, top - 10), text_label, fill="red", font=font)
                
                video_folder_path = video_path.parent
                image_name        = video_path.stem + "_first.jpg"
                detected_frame_path = video_folder_path / image_name
                img.save(detected_frame_path)

                person_timestamp = frame_idx/fps
                break

        frame_idx += 1

    cap.release()
    return detected_frame_path,person_timestamp


def low_res_detection_and_capture_loop(pwd):

    # Open low-res capture
    logging.info(f"Opening low-res pipeline: {LOW_RES_PIPELINE}")
    cap = cv2.VideoCapture(LOW_RES_PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        logging.error("Failed to open low-res pipeline")
        return

    logging.info(f"Starting high pipeline as {HIGH_RES_PIPELINE}")

    hr_pipe_obj = Gst.parse_launch(HIGH_RES_PIPELINE)
    hr_pipe_obj.set_state(Gst.State.PLAYING)


    # small 2-frame deque logic to avoid false positives
    frame_q = deque(maxlen=2)

    time_start = time.perf_counter() # Time in seconds

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            frame_q.append(gray)

            if len(frame_q) < 2:
                continue

            prev_frame, curr_frame = frame_q[0], frame_q[1]
            delta = cv2.absdiff(prev_frame, curr_frame)
            thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_area = cv2.countNonZero(thresh)

            if motion_area > 200:    # tune threshold
                # small debounce: clear deque so we don't trigger repeatedly
                frame_q.clear()

                logging.info(f"Detected motion!")

                # frame is already the right dimension, let's just convert to 8bits
                input_data = np.expand_dims(np.array(frame, dtype=np.uint8), axis=0)

                boxes,classes,scores,num = run_tpu_inference(input_data)

                person_found        = False
                for i in range(num):
                    if (int(classes[i]) == person_idx) and (scores[i] > 0.25):
                        person_found        = True
                        logging.info(f"Person was in frame with confidence {scores[i]}")

                if person_found:

                    # Person detected -> capture event
                    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    logging.info(f"Person detected at {ts} — will wait {POST_BUFFER_SECS}s to capture post-buffer")

                    # wait for the post-buffer duration (during this time splitmuxsink keeps writing new segments)
                    time.sleep(POST_BUFFER_SECS)

                    # Copy those latest segments to event folder
                    event_folder = copy_segments_to_event(RAM_DIR, EVENTS_DIR, ts)

                    # notify stitch/send thread with the event folder path
                    event_queue.put(event_folder)

                    logging.info(f"Event folder queued for post-processing: {event_folder}")

                    frame_q.clear()

            # small sleep to yield — low-res pipeline already runs at 10fps via pipeline
            time.sleep(0.005)

            # Do cleanup: keep only the last 2minutes. 
            # There is a new segment every 10 seconds so no real need to check each loop iteration
            # Float comparison is fine, no need to fine grained precision
            if (time.perf_counter() - time_start ) > 10.0: 

                time_start = time.perf_counter()
                files = [f for f in RAM_DIR.iterdir() if f.is_file()]

                if len(files) > MAX_SEGMENTS:
                    # Sort by modification time, newest last
                    files.sort(key=os.path.getmtime)

                    # Delete oldest, keep newest max_segments
                    for f in files[:-MAX_SEGMENTS]:
                        try:
                            f.unlink()
                        except OSError as e:
                            logging.info(f"[prune] Failed to remove {f}: {e}")


    finally:
        cap.release()

def stitch_worker_thread():
    """
    Placeholder worker that receives event_folder paths and will stitch/send later.
    For now, it just logs the event folder; you can implement the stitch + Telegram upload logic here.
    """
    send_text("Starting recording")
    while True:
        folder = event_queue.get()
        logging.info(f"Stitch worker got event folder: {folder}")
        # TODO: stitch the MKV segments into a single file and send via Telegram

        video_path = stitch_segments(folder,folder)

        # Double check that a person is detected
        frame_with_person,person_timestamp = check_for_person(video_path)
        if frame_with_person != None:
            send_text(f"Found a person at {person_timestamp}s - {frame_with_person.as_posix()} - {video_path.as_posix()}")
            send_photo(frame_with_person.as_posix())
            send_video(video_path.as_posix())

        # For now just sleep to simulate work:
        time.sleep(2)
        logging.info(f"Stitch/send placeholder finished for {folder}")


# ----------------- Main -----------------
def main():
    logging.info("Program start")


    # Start stitch/send worker thread (will process completed event folders)
    stitch_thread = threading.Thread(target=stitch_worker_thread, name="StitchWorker", daemon=True)
    stitch_thread.start()

    # Run low-res detection + capture loop in main thread (or start as its own thread if you prefer)
    low_res_detection_and_capture_loop(RTSP_PWD)


if __name__ == "__main__":
    main()
