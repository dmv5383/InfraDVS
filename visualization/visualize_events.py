import os
import cv2
import numpy as np
from tqdm import tqdm  # Import tqdm for the progress bar

events_directory_1 = "datasets/data/Town10HD_MidRainSunset_1/infra_events_1"
images_directory = "datasets/data/Town10HD_MidRainSunset_1/infra_rgb_1"
frame_rate = 1000  # Configurable frame rate in frames per second
output_video_path = "output_video.mp4"  # Path to save the output video
write_to_video = True  # Flag to write to video instead of visualizing

# Get sorted lists of all .png files in both directories
png_files_1 = sorted([f for f in os.listdir(events_directory_1) if f.endswith('.png')])
png_files_2 = sorted([f for f in os.listdir(images_directory) if f.endswith('.png')])

# Initialize the previous image to a plain gray image
previous_image = np.full((720, 1280, 3), 128, dtype=np.uint8)  # Adjust the size as needed

# Initialize video writer if write_to_video is True
if write_to_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (2560, 720))  # Adjust the frame size as needed

# Loop through each event file and display the events and images
for event_file in tqdm(png_files_1, desc="Event Files", unit="file"):  # Add tqdm progress bar
    event_base = event_file.split('_')[0]
    event_path = os.path.join(events_directory_1, event_file)
    image_path = os.path.join(images_directory, f"{event_base}_img.png")

    event_image = cv2.imread(event_path)

    if event_image is None:
        print(f"Failed to load event image: {event_path}")
        continue

    if os.path.exists(image_path):
        rgb_image = cv2.imread(image_path)
        if rgb_image is None:
            print(f"Failed to load RGB image: {image_path}")
            continue
        previous_image = rgb_image

    # Concatenate images horizontally if previous_image exists, otherwise use event_image alone
    combined_image = np.hstack((event_image, previous_image))

    if write_to_video:
        video_writer.write(combined_image)
    else:
        cv2.imshow('Lidar Visualization', combined_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if write_to_video:
    video_writer.release()
else:
    cv2.destroyAllWindows()