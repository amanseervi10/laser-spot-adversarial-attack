import os
import argparse
import csv
from ultralytics import YOLO

def parse_args():
    """ Parse command-line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument("model_size", choices=["n", "s", "m", "l", "x"], help="YOLO model size")
    parser.add_argument("output_csv", help="CSV file to save results")
    parser.add_argument("--save_folder", help="Folder to save inferred images", default=None)
    return parser.parse_args()

def run_inference(model, image_paths, batch_size=10):
    """ Run inference in batches to avoid GPU overload """
    results = []
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i + batch_size]
        results.extend(model(batch, verbose=True))
    return results

def save_results(image_paths, results, output_csv):
    """ Save inference results in CSV """
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "inference_values"])
        for img, res in zip(image_paths, results):
            bboxes = []
            for box in res.boxes:
                if box.cls == 0: 
                    coords = box.xyxy[0].tolist()  
                    confidence = box.conf[0].item() 
                    bbox_str = f"({','.join(map(str, coords))} , {confidence})"
                    bboxes.append(bbox_str)
            writer.writerow([os.path.basename(img), " | ".join(bboxes)])

def main():
    """ Main function to handle inference and saving """
    
    args = parse_args()
    model = YOLO(f"yolov8{args.model_size}.pt")
    dataset_folder = "./datasets/dataset_curated_temp"
    image_paths = [os.path.join(dataset_folder, f) for f in os.listdir(dataset_folder) if f.endswith(".jpg")]
    results = run_inference(model, image_paths)
    save_results(image_paths, results, args.output_csv)
    
    if args.save_folder:
        os.makedirs(args.save_folder, exist_ok=True)
        for img, res in zip(image_paths, results):
            res.save(filename=os.path.join(args.save_folder, os.path.basename(img)))

if __name__ == "__main__":
    main()