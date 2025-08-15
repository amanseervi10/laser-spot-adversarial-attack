import numpy as np
import random
import cv2
import os
import sys
import time
import torch

sys.path.append(os.path.abspath("code"))
from laser_simulation.laser_simulation import add_laser_dot

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    :param box1: (x_min, y_min, x_max, y_max) for box 1
    :param box2: (x_min, y_min, x_max, y_max) for box 2
    :return: IoU value
    """
    x_min1, y_min1, x_max1, y_max1 = box1
    x_min2, y_min2, x_max2, y_max2 = box2

    # Calculate intersection
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    intersection_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

    # Calculate union
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0

def find_relevant_bbox(bboxes, prev_bbox):
    """
    Find the bounding box in the current iteration that has the highest IoU with the previous iteration's bbox.

    :param bboxes: List of bounding boxes [(x_min, y_min, x_max, y_max), ...] from the current iteration.
    :param prev_bbox: Bounding box (x_min, y_min, x_max, y_max) from the previous iteration.
    :return: The bounding box with the highest IoU in the current iteration.
    """
    best_iou = 0
    best_bbox = None

    for bbox in bboxes:
        iou = calculate_iou(prev_bbox, bbox)
        if iou > best_iou:
            best_iou = iou
            best_bbox = bbox

    return best_bbox


# def dot_optimization(image, yolo_model, dot_positions, bboxes, current_pedestrian_box, color_rgb, save_folder,
#                     num_dots=3, iterations=10, restarts=3, max_moves_per_dot=5, save_images=False):
#     """
#     Perform greedy optimization to find the best dot placements to reduce pedestrian detection confidence.

#     :param image: Input image.
#     :param yolo_model: YOLO model to infer bounding boxes.
#     :param dot_positions: List of all possible (x, y) positions.
#     :param bboxes: List of bounding boxes in the original image.
#     :param current_pedestrian_box: The bounding box of the pedestrian being attacked.
#     :param color_rgb: RGB color of the laser dot.
#     :param save_folder: Directory to save images with bounding boxes.
#     :param num_dots: Number of dots to place.
#     :param iterations: Number of optimization iterations.
#     :param restarts: Number of random restarts to avoid local minima.
#     :param max_moves_per_dot: Max different positions to test per dot (to reduce search space).
#     :return: The best dot positions and their effect on pedestrian confidence.
#     """

#     os.makedirs(save_folder, exist_ok=True)  # Ensure save folder exists
#     best_score = float('inf')  # Minimize pedestrian confidence
#     best_configuration = None
#     prev_bbox = current_pedestrian_box  # Track the previous bounding box
#     image_count = 0  # Counter for saving images

#     print("Starting dot optimization...")

#     for _ in range(restarts):
#         current_configuration = random.sample(dot_positions, num_dots)
#         modified_image =  add_laser_dot(image = image, color_rgb=color_rgb,dot_positions=current_configuration)
        
#         # Perform YOLO inference
#         results = yolo_model(modified_image)  
        
#         # Save image with detections
#         if save_images:
#             annotated_img = results[0].plot()  
#             cv2.imwrite(os.path.join(save_folder, f"infer_{image_count}.jpg"), annotated_img)
#             image_count += 1

#         new_bboxes = results[0].boxes.xyxy
#         confidences = results[0].boxes.conf
#         pedestrian_indices = (results[0].boxes.cls == 0).nonzero().squeeze()
#         new_pedestrian_count = len(pedestrian_indices)
#         print(f"Pedestrian count: {new_pedestrian_count}")
#         print(f"Bboxes length: {len(bboxes)}")

#         # Early stopping if pedestrian count decreases
#         if new_pedestrian_count < len(bboxes):  
#             return current_configuration, 0  

#         current_bbox = find_relevant_bbox(new_bboxes[pedestrian_indices], prev_bbox)

#         if current_bbox is not None:
#             current_score = confidences[pedestrian_indices][(new_bboxes[pedestrian_indices] == current_bbox).all(dim=1)].item()
#             print(f"Initial score: {current_score}")
#         else:
#             return current_configuration, 0       

#         improved = True
#         while improved:
#             improved = False
#             # print("here")
#             for i in range(num_dots):
#                 # print("here1")
#                 random.shuffle(dot_positions)
#                 test_positions = dot_positions[:max_moves_per_dot]

#                 for new_position in test_positions:
#                     # print("here2")
#                     new_configuration = current_configuration[:]
#                     new_configuration[i] = new_position

#                     modified_image = add_laser_dot(image = image, color_rgb=color_rgb,dot_positions=new_configuration)
#                     results = yolo_model(modified_image)  

#                     # Save image with detections
#                     if save_images:
#                         annotated_img = results[0].plot()
#                         cv2.imwrite(os.path.join(save_folder, f"infer_{image_count}.jpg"), annotated_img)
#                         image_count += 1

#                     new_bboxes = results[0].boxes.xyxy
#                     confidences = results[0].boxes.conf
#                     pedestrian_indices = (results[0].boxes.cls == 0).nonzero().squeeze()
#                     new_pedestrian_count = len(pedestrian_indices)

#                     if new_pedestrian_count < len(bboxes):  
#                         return new_configuration, 0  

#                     current_bbox = find_relevant_bbox(new_bboxes[pedestrian_indices], prev_bbox)

#                     if current_bbox is not None:
#                         new_score = confidences[pedestrian_indices][(new_bboxes[pedestrian_indices] == current_bbox).all(dim=1)].item()
#                     else:
#                         return new_configuration, 0  

#                     if new_score < current_score:
#                         current_configuration = new_configuration
#                         current_score = new_score
#                         prev_bbox = current_bbox
#                         improved = True

#         if current_score < best_score:
#             best_score = current_score
#             best_configuration = current_configuration

#     return best_configuration, best_score


def dot_optimization(image, yolo_model, dot_positions, bboxes, current_pedestrian_box, color_rgb, save_folder,
                    num_dots=3, iterations=10, restarts=3, max_moves_per_dot=5, save_images=False):
    """
    Perform greedy optimization to find the best dot placements to reduce pedestrian detection confidence.

    :param image: Input image.
    :param yolo_model: YOLO model to infer bounding boxes.
    :param dot_positions: List of all possible (x, y) positions.
    :param bboxes: List of bounding boxes in the original image.
    :param current_pedestrian_box: The bounding box of the pedestrian being attacked.
    :param color_rgb: RGB color of the laser dot.
    :param save_folder: Directory to save images with bounding boxes.
    :param num_dots: Number of dots to place.
    :param iterations: Number of optimization iterations.
    :param restarts: Number of random restarts to avoid local minima.
    :param max_moves_per_dot: Max different positions to test per dot (to reduce search space).
    :param save_images: Flag to save images.
    :param batch_size: Number of images per batch for inference.
    :return: The best dot positions and their effect on pedestrian confidence.
    """

    if save_images :
        os.makedirs(save_folder, exist_ok=True) 
    best_score = float('inf') 
    best_configuration = None
    prev_bbox = current_pedestrian_box  
    image_count = 0 
    # print("Starting optimization...")

    for _ in range(restarts):
        current_configuration = random.sample(dot_positions, num_dots)
        modified_image = add_laser_dot(image=image, color_rgb=color_rgb, dot_positions=current_configuration)

        results = yolo_model(modified_image,verbose = False)
        
        if save_images:
            annotated_img = results[0].plot()
            cv2.imwrite(os.path.join(save_folder, f"infer_{image_count}.jpg"), annotated_img)
            image_count += 1

        new_bboxes = results[0].boxes.xyxy
        confidences = results[0].boxes.conf
        class_labels = results[0].boxes.cls
        pedestrian_indices = torch.where(class_labels == 0)[0]
        if pedestrian_indices.numel() == 0:  
            return current_configuration, 0
        else:
            new_pedestrian_count = len(pedestrian_indices)  
            current_bbox = new_bboxes[pedestrian_indices]  
            current_score = confidences[pedestrian_indices].max().item() 

        # Early stopping if pedestrian count decreases
        if new_pedestrian_count < len(bboxes):  
            return current_configuration, 0  

        current_bbox = find_relevant_bbox(new_bboxes[pedestrian_indices], prev_bbox)

        if current_bbox is not None:
            current_score = confidences[pedestrian_indices][(new_bboxes[pedestrian_indices] == current_bbox).all(dim=1)].item()
        else:
            return current_configuration, 0        

        improved = True
        iteration_count = 0
        start = time.time()
        while improved and iteration_count<iterations:
            # print("Iteration", iteration_count)
            end = time.time()
            # if end - start > 8:
            #     break
            
            improved = False
            iteration_count += 1
            
            for i in range(num_dots):
                random.shuffle(dot_positions)
                test_positions = dot_positions[:max_moves_per_dot]

                batch_size = max_moves_per_dot

                batch_best_score = float('inf')
                batch_best_config = None
                batch_best_bbox = prev_bbox
                
                for j in range(0, len(test_positions), batch_size):
                    batch = test_positions[j:j + batch_size]
                    batch_configurations = []
                    batch_images = []
                    
                    for new_position in batch:
                        new_configuration = current_configuration[:]
                        new_configuration[i] = new_position
                        batch_images.append(add_laser_dot(image=image, color_rgb=color_rgb, dot_positions=new_configuration))
                        batch_configurations.append(new_configuration)
                    
                    results_batch = yolo_model(batch_images,verbose=False)  
                    
                    for k, results in enumerate(results_batch):
                        if save_images:
                            annotated_img = results.plot()
                            cv2.imwrite(os.path.join(save_folder, f"infer_{image_count}.jpg"), annotated_img)
                            image_count += 1
                        
                        new_bboxes = results.boxes.xyxy
                        confidences = results.boxes.conf
                        class_labels = results.boxes.cls
                        pedestrian_indices = torch.where(class_labels == 0)[0]
                        

                        if pedestrian_indices.numel() == 0:  # No pedestrians found
                            return batch_configurations[k], 0  
                        else:
                            new_pedestrian_count = len(pedestrian_indices)
                            current_bbox = new_bboxes[pedestrian_indices]
                            current_score = confidences[pedestrian_indices].max().item()

                        if new_pedestrian_count < len(bboxes):  
                            return batch_configurations[k], 0  

                        current_bbox = find_relevant_bbox(new_bboxes[pedestrian_indices], prev_bbox)

                        if current_bbox is not None:
                            new_score = confidences[pedestrian_indices][(new_bboxes[pedestrian_indices] == current_bbox).all(dim=1)].item()
                        else:
                            return batch_configurations[k], 0  

                        if new_score < batch_best_score:
                            batch_best_score = new_score
                            batch_best_config = batch_configurations[k]
                            batch_best_bbox = current_bbox

                    if batch_best_config is not None:
                        current_configuration = batch_best_config
                        current_score = batch_best_score
                        prev_bbox = batch_best_bbox
                        improved = True

        if current_score < best_score:
            best_score = current_score
            best_configuration = current_configuration

    return best_configuration, best_score