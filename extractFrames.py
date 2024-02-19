import cv2
import os

def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Initialize variables
    frame_count = 0

    # Loop through the video frames
    while True:
        # Read the next frame
        success, frame = video_capture.read()

        # If there are no more frames, break the loop
        if not success:
            break

        # Save the frame as an image
        frame_name = f"frame_{frame_count:04d}.jpg"
        frame_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(frame_path, frame)

        # Increment frame count
        frame_count += 1

    # Release the video capture object
    video_capture.release()

    print(f"{frame_count} frames extracted and saved to {output_folder}")

if __name__ == "__main__":
    video_path = "r.mp4"
    output_folder = "output"

    extract_frames(video_path, output_folder)
