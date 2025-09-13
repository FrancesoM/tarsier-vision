import numpy as np
from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import argparse
import time

def load_labels(path):
    labels = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            labels.append(l.strip("\n"))
    return labels

parser = argparse.ArgumentParser(description="Example script with TPU flag")
parser.add_argument(
    "--use-tpu",
    type=int,
    default=0,
    help="Whether to use the Edge TPU"
)
parser.add_argument(
    "--draw-boxes",
    type=int,
    default=0,
    help="Use a more complex model to also draw boxes")

args = parser.parse_args()

if args.use_tpu:
    print("TPU will be used")
    use_tpu = True
else:
    print("TPU will NOT be used")
    use_tpu = False

if args.draw_boxes: 
    print("Selecting more complex model to draw boxes, inference will be slower")
    model_cpu ="/models/efficientdet_lite0_320_ptq.tflite"
    model_tpu ="/models/efficientdet_lite0_320_ptq_edgetpu.tflite"
    labels    ="/models/coco_labels.txt"
else:
    print("Selecting fast model to only detect objects in the image")
    model_cpu="/models/tf2_mobilenet_v3_edgetpu_1.0_224_ptq.tflite"
    model_tpu="/models/tf2_mobilenet_v3_edgetpu_1.0_224_ptq_edgetpu.tflite"
    labels   ="/models/imagenet_labels.txt"

if use_tpu:

  print("Using TPU...")

  # Load the Edge TPU delegate
  #delegate = load_delegate('libedgetpu.so.1')
  print("creating delegate object")
  delegate = load_delegate('/usr/lib/aarch64-linux-gnu/libedgetpu.so.1')
  
  # Load the model
  print("loading the model")
  interpreter = Interpreter(model_path=model_tpu, experimental_delegates=[delegate])
  interpreter.allocate_tensors()
else:
  interpreter = Interpreter(model_path=model_cpu)
  interpreter.allocate_tensors()

# Load labels
labels_lookup = load_labels(labels)

# Get input details
print("get model details")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Using model with the following input details: {input_details[0]}")

# Find all images
image_dir = Path("./test_images")
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)  # create folder if it doesn't exist

# Remove all files in output_dir
for f in output_dir.iterdir():
    if f.is_file():
        f.unlink()

image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))

for image_path in image_files:
    img = Image.open(image_path).convert("RGB")
    in_shape = input_details[0]['shape']
    img_resized = img.resize((in_shape[2], in_shape[1]))  # width, height
    input_data = np.expand_dims(np.array(img_resized, dtype=np.uint8), axis=0)
    print(f"Working on {image_path}, resizing to {in_shape[2], in_shape[1]}")

    t0 = time.perf_counter()

    # Set input tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    t1 = time.perf_counter()
    print(f"======================= Inference took  {(t1 - t0)*1000:.2f} ms =======================")

    # Get output tensors

    if args.draw_boxes:

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]    # (num, 4)
        classes = interpreter.get_tensor(output_details[1]['index'])[0]  # (num,)
        scores = interpreter.get_tensor(output_details[2]['index'])[0]   # (num,)
        num = int(interpreter.get_tensor(output_details[3]['index'])[0])

        print(f"Detected {num} objects")

        classes_and_scores = list(zip(classes,labels_lookup,scores))
        classes_and_scores.sort(key=lambda x:x[2]) # Sort by scores

        # Print 10 highest scores
        for el in classes_and_scores[-10:]:
            print(f"{el[0]}:{el[1]} -> {el[2]:.2f}")


        # Draw detections on original image (not resized one)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        for i in range(num):
            if scores[i] > 0.3:
                ymin, xmin, ymax, xmax = boxes[i]
                (left, right, top, bottom) = (
                    xmin * img.width,
                    xmax * img.width,
                    ymin * img.height,
                    ymax * img.height
                )
                label = f"{labels_lookup[int(classes[i])]} {scores[i]:.2f}"
                print(f"Detected {label}, bbox=({left:.0f},{top:.0f},{right:.0f},{bottom:.0f})")

                # Draw box + label
                draw.rectangle([left, top, right, bottom], outline="red", width=3)
                draw.text((left, top - 10), label, fill="red", font=font)

        # Save annotated image
        img.save(output_dir / f"{image_path.stem}_labeled.jpg")
    
    else:

        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # shape (N,)

        # Get top-5 class indices
        top_k = output_data.argsort()[-5:][::-1]

        # Print top-5 predictions with scores
        for i in top_k:
            print(f"{labels_lookup[i]}: {output_data[i]:.3f}")

    
print("Test done")

# at the very end
interpreter = None
import gc
gc.collect()



