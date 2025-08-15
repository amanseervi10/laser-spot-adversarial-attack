import cv2
import glob
import re

# Define the path to your images
image_files = glob.glob("datasets/dot_optimization/081117.jpg/1/infer_*.jpg")

# Function to extract numerical part of filename
def extract_number(filename):
    match = re.search(r"infer_(\d+)", filename)
    return int(match.group(1)) if match else float("inf")  # Sort non-matching files last

# Sort images numerically based on "x" in infer_x.jpg
image_files = sorted(image_files, key=extract_number)

# Check if images are found
if not image_files:
    print("No images found in the specified directory.")
    exit()

# Read the first image to get dimensions
frame = cv2.imread(image_files[0])
h, w, _ = frame.shape

# Define the video writer (15 FPS)
fps = 15
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Write frames to video
for img in image_files:
    frame = cv2.imread(img)
    out.write(frame)

# Pause on the last frame for 5 seconds
last_frame = cv2.imread(image_files[-1])
for _ in range(fps * 5):  # 5 seconds at 15 FPS
    out.write(last_frame)

out.release()
cv2.destroyAllWindows()
print("Video saved as output.mp4")
