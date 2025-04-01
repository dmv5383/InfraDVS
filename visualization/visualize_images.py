import os
import cv2
import time
import json

images_directory = "visualization/test_data/data/Town10HD_ClearNoon_10/infra_rgb_2/"

# Get a sorted list of all .png files in the directory
png_files = sorted([f for f in os.listdir(images_directory) if f.endswith('.png')])

# Loop through each file and display the image
for png_file in png_files:
    png_path = os.path.join(images_directory, png_file)
    json_path = os.path.join(images_directory, png_file.replace('img.png', 'bboxes.json'))
    
    image = cv2.imread(png_path)
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if image is None:
        print(f"Failed to load image: {png_path}")
        continue

    # Load bounding box data from the corresponding JSON file
    if os.path.exists(json_path):
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)
            for annotation in data['annotations']:
                bbox = annotation['bbox_2d']
                x, y, w, h = bbox
                print(f"Bounding box: {bbox}")  # Debug print
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                category_id = annotation['category_id']
                category_name = next(cat['name'] for cat in data['categories'] if cat['id'] == category_id)
                print(f"Category name: {category_name}")  # Debug print
                cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 1)

    cv2.imshow('Lidar Visualization', image)

    # Wait for 100 ms before displaying the next image
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()