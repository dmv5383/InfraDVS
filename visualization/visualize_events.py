import os
import cv2
import numpy as np
import json
from tqdm import tqdm

events_directory_1 = "datasets/data/Town10HD_MidRainSunset_1/infra_events_1"
images_directory = "datasets/data/Town10HD_MidRainSunset_1/infra_rgb_1"

frame_rate = 1000  # Configurable frame rate in frames per second
output_video_path = "event_output_video.mp4"  # Path to save the output video
write_to_video = True  # Flag to write to video instead of visualizing

# Get sorted lists of all .png files in both directories
png_files_1 = sorted([f for f in os.listdir(events_directory_1) if f.endswith('.png')])
png_files_2 = sorted([f for f in os.listdir(images_directory) if f.endswith('.png')])

# Initialize the previous image to a plain gray image
# previous_image = np.full((720, 1280, 3), 128, dtype=np.uint8)  # Adjust the size as needed

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            try:
                data = json.load(json_file)
                for annotation in data['annotations']:
                    bbox = annotation['bbox_2d']
                    # bbox = annotation['bbox']
                    x, y, w, h = bbox
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    category_id = annotation['category_id']
                    category_name = next(cat['name'] for cat in data['categories'] if cat['id'] == category_id)
                    cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            except json.JSONDecodeError:
                print(f"Error decoding JSON file: {json_path}")
            except KeyError as e:
                print(f"Missing key in JSON file: {e} in {json_path}")
    return image

# Initialize video writer if write_to_video is True
if write_to_video:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (1280, 720))  # Adjust the frame size as needed

# Function to draw bounding boxes on an image
def draw_bounding_boxes(image, json_path):
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            try:
                data = json.load(json_file)
                for annotation in data['annotations']:
                    bbox = annotation['bbox_2d']
                    x, y, w, h = bbox
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    category_id = annotation['category_id']
                    category_name = next(cat['name'] for cat in data['categories'] if cat['id'] == category_id)
                    cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            except json.JSONDecodeError:
                print(f"Error decoding JSON file: {json_path}")
            except KeyError as e:
                print(f"Missing key in JSON file: {e} in {json_path}")
    return image

# Loop through each event file and display the events and images
for event_file in tqdm(png_files_1, desc="Event Files", unit="file"):  # Add tqdm progress bar
    event_base = event_file.split('_')[0]
    event_path = os.path.join(events_directory_1, event_file)
    event_json_path = os.path.join(events_directory_1, f"{event_base}_bboxes.json")

    # image_path = os.path.join(images_directory, f"{event_base}_img.png")
    # image_json_path = os.path.join(images_directory, f"{event_base}_bboxes.json")

    event_image = cv2.imread(event_path)

    if event_image is None:
        print(f"Failed to load event image: {event_path}")
        continue

    event_image = draw_bounding_boxes(event_image, event_json_path)

    # if os.path.exists(image_path):
    #     rgb_image = cv2.imread(image_path)
    #     if rgb_image is None:
    #         print(f"Failed to load RGB image: {image_path}")
    #         continue
    #     previous_image = rgb_image

    # Concatenate images horizontally if previous_image exists, otherwise use event_image alone
    # combined_image = np.hstack((event_image, previous_image))

    if write_to_video:
        video_writer.write(event_image)
    # else:
    #     cv2.imshow('Lidar Visualization', combined_image)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

if write_to_video:
    video_writer.release()
else:
    cv2.destroyAllWindows()