"""
This file aims for simulating the light by given prameters, including:
alpha: Initilized luminous intensity
beta: attenuation paramter, describe the illuminous changes with distance
wavelength (w): color of the light
light type (t): point light, tube light and area light
location (x,y,w): the position of the light sourcel

"""

from scipy import ndimage
import numpy as np   
import matplotlib.pyplot as plt
import math
from PIL import Image
import cv2
import random

def wavelength_to_rgb(wavelength, gamma=0.8):
    """
    Description:
    Given a wavelength in the range of (380nm, 750nm), visible light range.
    a tuple of intergers for (R,G,B) is returned. 
    The integers are scaled to the range (0, 1).
    
    Based on code: http://www.noah.org/wiki/Wavelength_to_RGB_in_Python

    Parameters:
        Wavelength: the given wavelength range in (380, 750) 
    Returns:
        (R,G,B): color range in (0,1)
    """
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma    
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0

    # Convert RGB values from (0,1) to (0,255)
    R = int(R * 255)
    G = int(G * 255)
    B = int(B * 255)
    return (R, G, B)   


def add_laser_dot(image, bbox=None, color_rgb=(0, 255, 0), dot_radius=5, num_dots=5, dot_positions=None):
    """
    Adds multiple realistic laser dots inside a bounding box on the image.
    Each laser dot has a bright center with a glowing aura.
    :param image: The input image.
    :param bbox: The bounding box in YOLO format [x_min, y_min, x_max, y_max].
    :param color_rgb: The color of the laser dots (default green).
    :param dot_radius: The maximum radius of the laser glow.
    :param num_dots: Number of laser dots to add.
    """

    height, width, _ = image.shape

    # Create a blank mask for the laser effect
    laser_mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Convert RGB color to BGR, since OpenCV uses BGR
    color_bgr = tuple(reversed(color_rgb))

    if(dot_positions is None):
        for _ in range(num_dots):
            # Randomly place the laser dot within the bounding box
            x_min, y_min, x_max, y_max = bbox
            x_dot = random.randint(x_min, x_max)
            y_dot = random.randint(y_min, y_max)

            # Draw a filled circle in the center with solid color
            cv2.circle(laser_mask, (x_dot, y_dot), dot_radius // 4, color_bgr, -1)

            # Add a glowing aura around the center using Gaussian blur
            cv2.circle(laser_mask, (x_dot, y_dot), dot_radius, color_bgr, -1)
    
    else:
        for dot in dot_positions:
            # Randomly place the laser dot within the bounding box
            x_dot = int(dot[0])
            y_dot = int(dot[1])

            # Draw a filled circle in the center with solid color
            cv2.circle(laser_mask, (x_dot, y_dot), dot_radius // 4, color_bgr, -1)

            # Add a glowing aura around the center using Gaussian blur
            cv2.circle(laser_mask, (x_dot, y_dot), dot_radius, color_bgr, -1)

    # Apply Gaussian blur for the entire mask to simulate light scattering
    laser_mask = cv2.GaussianBlur(laser_mask, (0, 0), sigmaX=dot_radius / 3, sigmaY=dot_radius / 3)

    # Blend the laser mask with the original image
    overlay = cv2.addWeighted(image, 1, laser_mask, 0.7, 0) 

    ## Below is a simpler version. It saves time, but is less realistic
    # alpha = 0.7
    # overlay = cv2.addWeighted(image, 1, laser_mask, alpha, 0)

    return overlay



# # Image path
# img_path = "datasets/dataset_curated/010041.jpg"

# # Load the image
# image = cv2.imread(img_path)

# # Check if the image is loaded successfully
# if image is None:
#     print(f"Error: Could not load image from {img_path}")
# else:
#     # Define a bounding box in YOLO format (replace with actual bbox values)
#     bbox = [817, 549, 938, 793]

#     # Add the laser dot to the image
#     modified_image = add_laser_dot(image, bbox, wavelength_to_rgb(400), dot_radius=5)   

#     # Save a copy of the modified image
#     # save_path = "datasets/laser.jpg"
#     # cv2.imwrite(save_path, modified_image)
#     # print(f"Modified image saved at: {save_path}")

#     # Optionally display the image
#     cv2.imshow("Modified Image", modified_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
