import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tflite_runtime.interpreter import Interpreter, load_delegate


def load_labels(path):
    labels = []
    with open(path, 'r') as f:
        for l in f.readlines():
            labels.append(l.strip())
    return labels


def main(args):
    # Paths
    wrk_dir = Path(args.wrk_dir)
    video_in = wrk_dir / args.video_in
    video_out = wrk_dir / f"{video_in.stem}_labeled.mkv"

    # Load labels
    labels_lookup = load_labels(args.labels)

    # Load TFLite model (CPU or TPU)
    if args.use_tpu:
        print(f"Loading EdgeTPU model: {args.model_tpu}")
        delegate = load_delegate("/usr/lib/aarch64-linux-gnu/libedgetpu.so.1")
        interpreter = Interpreter(
            model_path=args.model_tpu,
            experimental_delegates=[delegate]
        )
    else:
        print(f"Loading CPU model: {args.model_cpu}")
        interpreter = Interpreter(model_path=args.model_cpu)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    in_shape = input_details[0]['shape']  # [1, height, width, 3]

    # Open video
    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*'X264')  # mkv with h264 encoding
    out = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))

    font = ImageFont.load_default()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Prepare input tensor
        img_resized = img.resize((in_shape[2], in_shape[1]))
        input_data = np.expand_dims(np.array(img_resized, dtype=np.uint8), axis=0)

        # Inference
        t0 = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        t1 = time.perf_counter()

        print(f"[Frame {frame_idx}] Inference took {(t1 - t0) * 1000:.2f} ms")

        # Get outputs
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        num = int(interpreter.get_tensor(output_details[3]['index'])[0])

        draw = ImageDraw.Draw(img)

        for i in range(num):
            if scores[i] > args.score_thr:
                ymin, xmin, ymax, xmax = boxes[i]
                (left, right, top, bottom) = (
                    xmin * img.width,
                    xmax * img.width,
                    ymin * img.height,
                    ymax * img.height,
                )
                label = f"{labels_lookup[int(classes[i])]} {scores[i]:.2f}"
                print(f"   {label}, bbox=({left:.0f},{top:.0f},{right:.0f},{bottom:.0f})")

                draw.rectangle([left, top, right, bottom], outline="red", width=3)
                draw.text((left, max(0, top - 10)), label, fill="red", font=font)

        # Convert back to OpenCV BGR
        frame_out = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        out.write(frame_out)

    cap.release()
    out.release()
    print(f"Saved labeled video to {video_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wrk_dir", type=str, required=True, help="Working directory containing video")
    parser.add_argument("--video_in", type=str, required=True, help="Input video filename (mkv)")
    parser.add_argument("--model_cpu", type=str, default="/models/efficientdet_lite0_320_ptq.tflite")
    parser.add_argument("--model_tpu", type=str, default="/models/efficientdet_lite0_320_ptq_edgetpu.tflite")
    parser.add_argument("--labels", type=str, default="/models/coco_labels.txt")
    parser.add_argument("--use_tpu", action="store_true", help="Use EdgeTPU delegate")
    parser.add_argument("--score_thr", type=float, default=0.3, help="Score threshold for detections")
    args = parser.parse_args()

    main(args)
