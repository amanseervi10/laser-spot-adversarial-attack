import os
import pandas as pd
from ultralytics import YOLO  
from simulate_attack import simulate_attack
import time

# Load the YOLO model
model = YOLO("yolov8n.pt")

def log_to_file(message, file="analysis.txt"):
    with open(file, "a") as f: 
        f.write(message + "\n")

def pedestrian_count_csv(csv_file):
    previous_pedestrian_count = 0
    csv_image_counts = {}

    # Read the CSV file
    data = pd.read_csv(csv_file)

    for _, row in data.iterrows():
        image_name = row['image_name']
        inference_values = row['inference_values']
        
        # Skip empty rows
        if pd.isna(inference_values) or inference_values.strip() == "":
            csv_image_counts[image_name] = 0
            continue
        
        # Split and count inference values
        inferences = inference_values.split(" | ")
        pedestrian_count = len(inferences)  # Each tuple represents a pedestrian
        previous_pedestrian_count += pedestrian_count
        csv_image_counts[image_name] = pedestrian_count
    
    return previous_pedestrian_count

# Function to detect pedestrians in an image
def count_pedestrians_yolo(image_path):
    results = model(image_path,device=0)
    pedestrian_count = 0
    for result in results:
        for box in result.boxes.data:  
            cls = int(box[5]) 
            if cls == 0: 
                pedestrian_count += 1
    return pedestrian_count

def temp(num_dots,wavelength, images_folder, inference_csv, output_folder,previous_pedestrian_count):

    start_time = time.time()

    simulate_attack(wavelength,num_dots,images_folder,inference_csv,output_folder)

    # 1. Count pedestrians detected now using YOLO
    current_pedestrian_count = 0
    image_counts = {}  # Store pedestrian counts for each image

    for image_name in os.listdir(output_folder):
        image_path = os.path.join(output_folder, image_name)
        pedestrian_count = count_pedestrians_yolo(image_path)
        current_pedestrian_count += pedestrian_count
        image_counts[image_name] = pedestrian_count

    # 3. Calculate attack success rate
    attack_success_rate = (previous_pedestrian_count - current_pedestrian_count) / previous_pedestrian_count

    end = time.time()

    # 4. Print results
    log_to_file(f"\nTotal pedestrians detected previously: {previous_pedestrian_count}", file=f"analysis_{wavelength}.txt")
    log_to_file(f"Total pedestrians detected now : {current_pedestrian_count}",file=f"analysis_{wavelength}.txt")
    log_to_file(f"Attack Success Rate: {attack_success_rate:.2%}",file=f"analysis_{wavelength}.txt")
    log_to_file(f"Time taken: {end - start_time:.2f} seconds",file=f"analysis_{wavelength}.txt")

    return attack_success_rate


def analyze_attack(start_dots, end_dots, iteration_count,wavelength, images_folder, inference_csv, output_folder,previous_pedestrian_count):
    for i in range(start_dots, end_dots + 1):
        cumulative_asr = 0

        # Print the num_dots for which we are currently doing
        log_to_file("------------------------------------------------------------",file=f"analysis_{wavelength}.txt")
        log_to_file(f"Num Dots: {i}",file=f"analysis_{wavelength}.txt")
        log_to_file("------------------------------------------------------------",file=f"analysis_{wavelength}.txt")

        start = time.time()

        for j in range(iteration_count):
            cumulative_asr += temp(i,wavelength, images_folder, inference_csv, output_folder,previous_pedestrian_count)

        end = time.time()
        
        log_to_file(f"\nAverage ASR for {i} dots: {cumulative_asr / iteration_count}",file=f"analysis_{wavelength}.txt")
        log_to_file(f"Time taken all iterations for {i} dots: {end - start:.2f} seconds\n",file=f"analysis_{wavelength}.txt")


if __name__ == "__main__":

    wavelengths = [380,480,532,580,680]
    # wavelengths = [532]
    image_dir = "datasets/dataset_curated_temp" 
    csv_file = "inference_nano.csv" 
    output_folder = "datasets/laser_simulated"

    previous_pedestrian_count = pedestrian_count_csv(csv_file)
    print(previous_pedestrian_count)

    for wavelength in wavelengths:

        log_to_file(f"Analysis for Wavelength {wavelength}\n\n",file=f"analysis_{wavelength}.txt")

        start = time.time()
        analyze_attack(2, 5, 5, wavelength, image_dir, csv_file, output_folder,previous_pedestrian_count)
        end = time.time()

        log_to_file(f"Time taken for analysis across all iterations: {end - start:.2f} seconds",file=f"analysis_{wavelength}.txt")
    
    end = time.time()