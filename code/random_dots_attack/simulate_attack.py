import os
import cv2
import pandas as pd
import sys

sys.path.append(os.path.abspath("code"))
from laser_simulation.laser_simulation import *

def simulate_attack(laser_wavelength,num_dots,images_folder,inference_csv,output_folder):

    #deciding on the color
    color_rgb = wavelength_to_rgb(laser_wavelength)
    print(f"Color RGB: {color_rgb}")

    os.makedirs(output_folder, exist_ok=True)

    inference_df = pd.read_csv(inference_csv)

    # Loop through each row in the CSV
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

        # Process bounding boxes
        for bbox_str in bboxes.split(" | "):
            try:
                bbox = eval(bbox_str)
                x1, y1, x2, y2, confidence = bbox

                #convert each into integer
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                #simulating the laser
                image = add_laser_dot(image, (x1, y1, x2, y2), color_rgb, dot_radius=5, num_dots=num_dots)
                print(f"Added laser dot to {image_name} at ({x1}, {y1}, {x2}, {y2}).")
                

            except Exception as e:
                print(f"Error processing bbox '{bbox_str}' in {image_name}: {e}")
                continue
        

        output_image_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_image_path, image)
        print(f"Processed {image_name} and saved with laser dots.")

    print("All images processed and saved with laser dots.")    


if __name__ == "__main__":
    
    images_folder = 'datasets/dataset_curated_temp' 
    inference_csv = 'inference_nano.csv'  
    output_folder = 'datasets/laser_simulated'

    simulate_attack(532,3,images_folder,inference_csv,output_folder)