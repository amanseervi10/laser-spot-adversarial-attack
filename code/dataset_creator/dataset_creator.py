"""
This is to create a random dataset of 30 images from 15 of the different locations in the LLVIP dataset.
We make sure that each image selected has at least one pedestrian in it.

To run : python .\code\dataset_creator.py

"""

import os
import random
import shutil
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

dataset_folder = "datasets/LLVIP/visible/train"
dataset_curated = "datasets/dataset_curated_temp"
if not os.path.exists(dataset_curated):
    os.makedirs(dataset_curated)

map = {}

for file in os.listdir(dataset_folder):
    '''
    Such is the nomenclature of the LLVIP dataset, that the first 2 characters 
    in the file name can be used to distinguish between different scenes. So 
    we use that as they key in our map and the values are the list of images
    from that scene
    '''
    if file.endswith(".jpg"):
        if file[0:2] in map:
            map[file[0:2]].append(file)
        else:
            map[file[0:2]] = [file]


def filter_images_with_people(images, batch_size=20, max_valid=30):
    """ Run batch inference in smaller batches to avoid GPU overload. Stop when we have enough valid images (i.e 30). """

    valid_images = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        image_paths = [os.path.join(dataset_folder, img) for img in batch]
        results = model(image_paths, verbose=True) 
        valid_images.extend([img for img, res in zip(batch, results) if any(box.cls == 0 for box in res.boxes)])
        if len(valid_images) >= max_valid:
            return valid_images[:max_valid]
    return valid_images

for i in range(1, 16):
    idx  = str(i)
    if len(idx) == 1:
        idx = "0" + idx
    images = map[str(idx)]

    random.shuffle(images)
    images = filter_images_with_people(images)
    images = images[:30]

    for image in images:
        shutil.copy(os.path.join(dataset_folder, image), os.path.join(dataset_curated, image))