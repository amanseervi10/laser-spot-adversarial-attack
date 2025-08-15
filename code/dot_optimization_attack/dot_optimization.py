import os
import sys
import pandas as pd
import cv2
import time
from tqdm import tqdm
import argparse
from ultralytics import YOLO
from optimization_helper import dot_optimization
from analyse_attack import parse_stats

sys.path.append(os.path.abspath("code"))
from laser_simulation.laser_simulation import *

# Used for profiling code
import cProfile
import pstats
import io

'''
python .\code\dot_optimization_attack\dot_optimization.py --model n --inference_csv datasets/inference_results.csv --images_folder datasets/dataset_curated --wavelength 532 --num_dots 3 --iterations 10 --restarts 3 --moves_per_dot 5 --save_image datasets/dot_optimization_temp --log log_file.txt --stats stats.txt


python .\code\dot_optimization_attack\dot_optimization.py --model s --inference_csv inference_small.csv --images_folder datasets/dataset_curated_temp --wavelength 532 --num_dots 3 --iterations 5 --restarts 3 --moves_per_dot 5 --log log_file_overleaf.txt --stats stats.txt --save_images for_overleaf
'''

def parse_args():
    """ Parse command-line arguments. """
    parser = argparse.ArgumentParser(description="Adversarial Attack on Object Detection Model")

    parser.add_argument("--model", choices=["n", "s", "m"], required=True, help="YOLO model size: nano (n), small (s), or medium (m)")
    parser.add_argument("--inference_csv", type=str, required=True, help="Path to CSV file with inference results")
    parser.add_argument("--images_folder", type=str, required=True,help="Path to the folder containing input images")
    parser.add_argument("--wavelength", type=int, choices=range(380, 751), metavar="[380-750]", required=True, help="Wavelength of the laser (between 380 and 750 nm)")
    parser.add_argument("--num_dots", type=int, required=True, help="Number of laser dots to apply")
    parser.add_argument("--iterations", type=int, required=True, help="Number of iterations for optimization")
    parser.add_argument("--restarts", type=int, required=True, help="Number of restarts for attack optimization")
    parser.add_argument("--moves_per_dot",type=int, required=True, help="Number of moves per dot that we check when optimizing")
    parser.add_argument("--save_image", type=str, help="Folder path to save attacked images (optional)")
    parser.add_argument("--log", type=str, help="File to log individual pedestrian results (optional)")
    parser.add_argument("--stats", type=str, help="File to save final attack statistics (optional)")
    parser.add_argument("--profile", action="store_true", help="Enable profiling if specified")

    args = parser.parse_args()

    if args.stats and not args.log:
        parser.error("--stats requires --log to be specified.")

    return args


def log_to_file(message, filename="log_file.txt"):
    with open(filename, "a") as f:  
        f.write(str(message) + "\n")

def get_dot_positions(bbox, min_cell_width=10, min_cell_height=10):
    """
    Generates all possible dot coordinates within the bounding box based on grid cells.

    :param bbox: The bounding box in [x_min, y_min, x_max, y_max] format.
    :param min_cell_width: The minimum width of each grid cell.
    :param min_cell_height: The minimum height of each grid cell.
    :return: A list of (x, y) coordinates for possible dot positions.
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
        for col in range(int(grid_cols)):
            x = int(x_min + col * cell_width + cell_width / 2)
            y = int(y_min + row * cell_height + cell_height / 2)
            dot_positions.append((x, y))

    return dot_positions


def simulate_dot_optimization_attack(args):

    yolo_model = YOLO(f"yolov8{args.model}.pt")
    inference_df = pd.read_csv(args.inference_csv)
    color_rgb = wavelength_to_rgb(args.wavelength)

    if args.save_image:
        os.makedirs(args.save_image, exist_ok=True)

    count_fooled = 0
    total_count = 0
    count_temp = 0
    for _, row in tqdm(inference_df.iterrows(), total=len(inference_df), desc="Processing images"):

        # count_temp += 1
        # if count_temp == 2: 
        #     break
        image_name, bboxes = row['image_name'], row['inference_values']
        if pd.isna(bboxes) or not bboxes.strip():
            continue

        image_path = os.path.join(args.images_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            continue

        if args.save_image:
            os.makedirs(f"{args.save_image}/{image_name}", exist_ok=True)

        bbox_count = 0
        # print("Image name : ", image_path)
        for bbox_str in bboxes.split(" | "):
            try:
                bbox = eval(bbox_str)
                x1, y1, x2, y2, _ = map(int, bbox)
                dot_positions = get_dot_positions([x1, y1, x2, y2])
                
                save_path = f"{args.save_image}/{image_name}/{bbox_count}" if args.save_image else None
                start = time.time()
                best_config, best_score = dot_optimization(image, yolo_model, dot_positions, bboxes.split(" | "),
                                                            [x1, y1, x2, y2], color_rgb, save_path,
                                                            args.num_dots,args.iterations, args.restarts,
                                                            args.moves_per_dot,save_images=args.save_image is not None,)
                
                # print("Best configuration: ", best_config, "Best score: ", best_score)
                
                
                end = time.time()
                
                if args.log:
                    log_to_file(f"Processed {image_name} bbox {bbox_str}", args.log)
                    log_to_file(f"Number of position: {len(dot_positions)}", args.log)
                    log_to_file(f"Best configuration: {best_config}, Best score: {best_score}", args.log)
                    log_to_file(f"Time taken: {end - start:.2f}s\n", args.log)
                
                if best_config and best_score == 0:
                    count_fooled += 1
                bbox_count += 1
            except Exception as e:
                log_to_file(f"Error processing bbox {bbox_str}: {e}","error_log.txt")
                continue
            
        total_count += bbox_count
    
    log_to_file(f"Attack success rate: {count_fooled / total_count * 100:.2f}%", args.log)

    if args.stats:
        parse_stats(args)


def main() :
    
    start = time.time()
    args = parse_args()

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        start = time.time()

        simulate_dot_optimization_attack(args)

        profiler.disable()
        stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stream).sort_stats("tottime")
        stats.print_stats(5)
        print(stream.getvalue())
    
    else : 
        simulate_dot_optimization_attack(args)
    
    end = time.time()
    print(f"Time taken: {end - start:.2f} seconds")


if __name__ == "__main__":
    main()