#Read from a txt file which contains paths to the scenes
#We are going to select random indices from the cameras
#Make train.py read this indices each time
#Calls train accordinly 5 times for each scene


import numpy as np
import subprocess
import sys
import os
from tqdm import tqdm
from random import sample

if __name__ == "__main__":
	#subprocess.run("rm -r results", shell = True)
	with open('sources.txt', 'r') as file:
	    # Read lines from the file
	    sources = file.readlines()
	sources = [source.strip() for source in sources]
	
	#Create folder for outputs
	os.makedirs('results', exist_ok = True)

	for path in tqdm(sources, desc="Processing sources"):
		#Make path for saving
		parts = path.split(os.path.sep)
		name = parts[-1]

		for n_cam in [8,10]: #[3, 4, 5, 8, 10]:
			for i in range(5):
				#Get random cameras
				imgs = range(len(os.listdir(os.path.join(path, "images"))))
				cams_ids = sample(imgs, n_cam)
				cams_ids.sort()
				cameras = ' '.join(map(str, cams_ids))

				#Train model with depth normalization
				output_path = './results/' + name  + "/" + str(n_cam) + "_cams/depth/" + str(i)
				command = "python3 train.py -s {} -m {} --sh_degree 1 --opacity_reset_interval 100000000 --iterations 30000 --save_iterations 30000 --cameras {} --depths true --TVL 0.1".format(path, output_path, cameras)
				subprocess.run(command, shell=True)
				#Gets metrics
				command = "python3 render.py -m " + output_path
				subprocess.run(command, shell=True)
				command = "python3 metrics.py -m " + output_path
				subprocess.run(command, shell=True)

				#Train model with no depth normalization
				output_path = './results/' + name  + "/" + str(n_cam) + "_cams/no_depth/" + str(i)
				command = "python3 train.py -s {} -m {} --sh_degree 1 --opacity_reset_interval 100000000 --iterations 30000 --save_iterations 30000 --cameras {} --depths false".format(path, output_path, cameras)
				subprocess.run(command, shell=True)
				#Metrics
				command = "python3 render.py -m " + output_path
				subprocess.run(command, shell=True)
				command = "python3 metrics.py -m" + output_path
				subprocess.run(command, shell=True)
				
