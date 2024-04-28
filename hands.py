import cv2
from ultralytics import YOLO
import threading
import os
import time
from collections import Counter
from PIL import Image

def keep_one_image(folder_path):
  """
  Deletes all but one image in a folder.

  Args:
      folder_path (str): Path to the folder containing images.
  """
  # Skip deletion if there's only one file (or less)
  num_files = len(os.listdir(folder_path))
  if num_files <= 1:
    print(f"Folder already contains {num_files} image(s). Skipping deletion.")
    return

  # Track the first encountered non-deleted image
  first_kept_image = None
  for filename in os.listdir(folder_path):
    # Construct full image path
    image_path = os.path.join(folder_path, filename)
    # Check if it's a file (not a directory) and ends with a common image extension
    if os.path.isfile(image_path) and image_path.lower().endswith((".jpg", ".jpeg", ".png")):
      # Delete the image
      os.remove(image_path)
      print(f"Image deleted: {image_path}")
      # Keep track of the first non-deleted image
      if first_kept_image is None:
        first_kept_image = filename

  # Print message if no images were kept (all deleted)
  if first_kept_image is None:
    print(f"All images in folder were deleted. No images kept.")


def delete_if_dominant_color(image_path, target_color):
  """
  Opens an image, checks for dominant color, and deletes it if it matches.

  Args:
      image_path (str): Path to the image file.
      target_color (tuple): RGB tuple representing the color to check for deletion (e.g., (255, 0, 0) for red).
  """
  try:
    # Open the image
    image = Image.open(image_path)
    pixels = image.load()

    # Get image dimensions
    width, height = image.size

    # Count color occurrences
    color_counts = Counter()
    for x in range(width):
      for y in range(height):
        color_counts[pixels[x, y]] += 1

    # Find the most frequent color
    dominant_color = color_counts.most_common(1)[0][0]

    # Check if dominant color matches target color and delete if so
    if dominant_color == target_color:
      os.remove(image_path)
      print(f"Image deleted: {image_path} (Dominant color: {dominant_color})")
    else:
      print(f"Image kept: {image_path} (Dominant color: {dominant_color})")
  except OSError:
    print(f"Error opening image: {image_path}")

def iterate_and_delete(folder_path, target_color):
  """
  Iterates over a folder and calls delete_if_dominant_color for each image.

  Args:
      folder_path (str): Path to the folder containing images.
      target_color (tuple): RGB tuple representing the color to check for deletion.
  """
  for filename in os.listdir(folder_path):
    # Construct full image path
    image_path = os.path.join(folder_path, filename)
    # Check if it's a file (not a directory) and ends with a common image extension
    if os.path.isfile(image_path) and image_path.lower().endswith((".jpg", ".jpeg", ".png")):
      delete_if_dominant_color(image_path, target_color)

# Example usage

def capture_image(camera_index, folder_path):
  """
  Captures a single frame from the specified camera and saves it to the folder.

  Args:
      camera_index (int): The index of the camera to capture from.
      folder_path (str): The path to the folder where the image will be saved.
  """

  # Open the specified camera
  cap = cv2.VideoCapture(camera_index)

  # Check if camera opened successfully
  if not cap.isOpened():
      print(f"Error opening camera {camera_index}")
      return False

  # Capture a single frame
  ret, frame = cap.read()

  # Check if frame capture was successful
  if not ret:
      print(f"Failed to capture frame from camera {camera_index}")
      cap.release()
      cv2.destroyAllWindows()
      return False

  # Generate a unique identifier (optional)
  timestamp = str(int(time.time()))

  # Construct filename with folder path (without extension)
  filename = os.path.join(folder_path, f"captured_image_{timestamp}")  # Example filename

  # Save the image automatically using folder path and generated identifier (optional)
  cv2.imwrite(filename + ".jpg", frame)  # Add extension here
  print(f"Image from camera {camera_index} saved to: {folder_path}")

  # Release the camera (important for closing the camera)
  cap.release()
  cv2.destroyAllWindows()  # Close any open windows (might not be necessary)

  return True  # Indicate successful capture

def handle_detections(results, folder_path, secondary_camera_index, target_color):
    """
    Processes detections and executes commands based on findings.

    Args:
        results: The results from the YOLO model prediction.
        folder_path (str): The path to the folder where images are saved.
        secondary_camera_index (int): The index of the secondary camera to use.
    """

    for r in results:
        boxes = r.boxes.xyxy  # Get bounding boxes
        if len(boxes) > 0:  # Check if there are detections
            # Object detected!
            print(f"Object detected: {r.names[0]}")
            # You can extend this to execute specific commands based on the object label
            # execute_command(f"command_for_{r.names[0]}")

            # Close the primary camera explicitly
            for i in range(10):
                time.sleep(1)
                capture_image(secondary_camera_index, folder_path)
            iterate_and_delete(folder_path, target_color)
            keep_one_image(folder_path)

            break  # Exit loop after capturing on detection

    return  # No need to capture image here (optional in handle_detections)


# Load YOLO model
model = YOLO('best (7).pt')

# Confidence threshold for detection
conf_threshold = 0.5

# Example usage: folder path for captured images and secondary camera index
folder_path = "images_folder"
secondary_camera_index = 0  # Modify this to your secondary camera index
target_color = (255, 137, 37)
# Run inference on primary camera (modify index if needed)
results = model.predict(source="1", save=False, imgsz=640, conf=conf_threshold, show=True, stream=True)
thread = threading.Thread(target=handle_detections(results, folder_path, secondary_camera_index,target_color), args=(False,))

# Process detections
thread.start

