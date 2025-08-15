import subprocess
import os

commands = [
    # r'python code\dot_optimization_attack\dot_optimization.py --model n --inference_csv inference_nano.csv --images_folder datasets/dataset_curated_temp --wavelength 532 --num_dots 3 --iterations 5 --restarts 1 --moves_per_dot 5 --log log_file_restarts_1.txt --stats stats.txt',
    # r'python code\dot_optimization_attack\dot_optimization.py --model n --inference_csv inference_nano.csv --images_folder datasets/dataset_curated_temp --wavelength 532 --num_dots 3 --iterations 5 --restarts 3 --moves_per_dot 5 --log log_file_restarts_3.txt --stats stats.txt',
    # r'python code\dot_optimization_attack\dot_optimization.py --model n --inference_csv inference_nano.csv --images_folder datasets/dataset_curated_temp --wavelength 532 --num_dots 3 --iterations 5 --restarts 5 --moves_per_dot 5 --log log_file_restarts_5.txt --stats stats.txt',
    # r'python code\dot_optimization_attack\dot_optimization.py --model n --inference_csv inference_nano.csv --images_folder datasets/dataset_curated_temp --wavelength 532 --num_dots 3 --iterations 5 --restarts 10 --moves_per_dot 5 --log log_file_restarts_10.txt --stats stats.txt',
    r'python code\dot_optimization_attack\dot_optimization.py --model n --inference_csv inference_nano.csv --images_folder datasets/dataset_curated_temp --wavelength 532 --num_dots 3 --iterations 5 --restarts 20 --moves_per_dot 5 --log log_file_restarts_20.txt --stats stats.txt'
]

for cmd in commands:
    process = subprocess.run(cmd, shell=True, check=True)
    print(f"Finished: {cmd}")
