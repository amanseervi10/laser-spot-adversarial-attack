import os
import pandas as pd
import cv2
from laser_simulation.laser_simulation import *


def get_consecutive_dot_positions_dynamic(bbox, min_cell_width=20, min_cell_height=20):
    """
    Generates 3 consecutive dot coordinates within the bounding box with dynamically calculated grid rows and columns.
    The bounding box is divided into grids based on the minimum cell size.

    :param bbox: The bounding box in [x_min, y_min, x_max, y_max] format.
    :param min_cell_width: The minimum width of each grid cell.
    :param min_cell_height: The minimum height of each grid cell.
    :return: A list of tuples, where each tuple contains 3 (x, y) coordinates for the consecutive dots.
    """
    x_min, y_min, x_max, y_max = bbox

    # Calculate the bounding box width and height
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    # Calculate dynamic number of rows and columns based on minimum cell size
    grid_cols = max(3, bbox_width // min_cell_width)  
    grid_rows = max(1, bbox_height // min_cell_height)

    # Calculate the actual cell size
    cell_width = bbox_width / grid_cols
    cell_height = bbox_height / grid_rows

    dot_positions = []

    # Iterate over the grid cells
    for row in range(int(grid_rows)):
        for col in range(int(grid_cols) - 2):                            
            x1 = int(x_min + col * cell_width + cell_width / 2)
            y1 = int(y_min + row * cell_height + cell_height / 2)
            
            x2 = int(x1 + cell_width)
            y2 = y1
            
            x3 = int(x2 + cell_width)
            y3 = y1

            dot_positions.append(((x1, y1), (x2, y2), (x3, y3)))

    return dot_positions


def simulate_consecutive_dots_attack(laser_wavelength):

    #deciding on the color
    color_rgb = wavelength_to_rgb(laser_wavelength)
    print(f"Color RGB: {color_rgb}")

    images_folder = 'datasets/dataset_curated' 
    inference_csv = 'datasets/inference_results.csv'  
    output_folder = 'datasets/consecutive_dots_simulated'

    os.makedirs(output_folder, exist_ok=True)

    inference_df = pd.read_csv(inference_csv)

    for _, row in inference_df.iterrows():
        image_name = row['image_name']
        bboxes = row['inference_values']


        if pd.isna(bboxes) or not bboxes.strip():
            continue

        image_path = os.path.join(images_folder, image_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Warning: Could not load image {image_name}. Skipping.")
            continue

        os.makedirs(output_folder+f"/{image_name}", exist_ok=True)

        bbox_count = 0
        for bbox_str in bboxes.split(" | "):
            try:
                bbox = eval(bbox_str)
                x1, y1, x2, y2, confidence = bbox

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                dot_positions = get_consecutive_dot_positions_dynamic([x1, y1, x2, y2])

                os.makedirs(output_folder+f"/{image_name}/"+str(bbox_count), exist_ok=True)

                count = 0
                for dots in dot_positions:
                    image_temp = image
                    image_temp =  add_laser_dot(image_temp, (x1, y1, x2, y2), color_rgb, dot_radius=5, num_dots=3, dot_positions=dots)

                    output_image_path = os.path.join(output_folder+f"/{image_name}"+"/"+str(bbox_count), f"{count}.jpg")
                    cv2.imwrite(output_image_path, image_temp)
                    count += 1

                bbox_count += 1

            except Exception as e:
                print(f"Error processing bbox '{bbox_str}' in {image_name}: {e}")
                continue

    print("All images processed and saved with laser dots.")    


def attack_success():
    import glob
    import shutil
    import time
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")

    start = time.time()

    original_count = 681
    inference_csv = 'datasets/inference_results.csv'
    output_folder = 'datasets/consecutive_dots_simulated'
    save_folder = 'datasets/successful_attack_images'


    inference_df = pd.read_csv(inference_csv)

    successful_attack_count = 0

    for _, row in inference_df.iterrows():
        image_name = row['image_name']
        bboxes = row['inference_values']

        if pd.isna(bboxes) or not bboxes.strip():
            continue

        bbox_count_csv = len(bboxes.split(" | "))

        image_folder_path = os.path.join(output_folder, image_name)

        if not os.path.exists(image_folder_path):
            print(f"Folder not found for {image_name}. Skipping.")
            continue

        bbox_folders = glob.glob(os.path.join(image_folder_path, "*"))
        for bbox_folder in bbox_folders:

            simulated_images = glob.glob(os.path.join(bbox_folder, "*.jpg"))
            for sim_image_path in simulated_images:

                results = model.predict(source=sim_image_path, save=False, save_txt=False, save_conf=False)

                pedestrian_count = 0
                for box in results[0].boxes: 
                    class_id = int(box.cls[0].item())  
                    if class_id == 0:
                        pedestrian_count += 1

                if pedestrian_count == bbox_count_csv - 1:
                    successful_attack_count += 1

                    # run the model again on this one, this time with bounding box thing and then save in the save_folder
                    prediction_results = model.predict(
                        source=sim_image_path,
                        save=True,  # Save images with bounding boxes
                        save_txt=False,  # Don't save text files
                        save_conf=False,  # Don't save confidence values
                        project="temp_results",  # Temporary folder for saving
                        name="bbox_images"  # Subfolder to isolate results
                    )

                    # Move and rename the saved image
                    temp_folder = os.path.join("temp_results", "bbox_images")
                    saved_image_path = os.path.join(temp_folder, os.path.basename(sim_image_path))

                    if os.path.exists(saved_image_path):
                        # Add random number suffix to the original image name
                        random_suffix = random.randint(1000, 9999)
                        new_image_name = f"{image_name}_attack_{random_suffix}.jpg"
                        destination_path = os.path.join(save_folder, new_image_name)

                        # Move the renamed image to the save folder
                        shutil.move(saved_image_path, destination_path)

                    # Clean up temporary folder
                    if os.path.exists(temp_folder):
                        shutil.rmtree(temp_folder)
                    
                    break  
    
    end = time.time()

    # Calculate attack success rate
    attack_success_rate = (successful_attack_count / original_count) * 100
    print(f"Total pedestrians detected previously: {original_count}")
    print(f"Total pedestrians detected now : {successful_attack_count}")
    print(f"Attack Success Rate: {attack_success_rate:.2f}%")
    print(f"Time taken: {end - start:.2f} seconds")


if __name__ == "__main__":
    # simulate_consecutive_dots_attack(532)
    attack_success()