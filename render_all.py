import os
import sys
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

import torch
import torchvision

from gaussian_renderer import render
from gaussian_renderer import GaussianModel

from scene import Scene
import os
from os import makedirs

from scene.cameras import MiniCam

#Se crea el parser
parser = ArgumentParser(description="Testing script parameters")

#Argumentos del modelo
model = ModelParams(parser)
#Se crea la clase pipeline y agregamos mas al parser
pipeline = PipelineParams(parser)

parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")

args = parser.parse_args(sys.argv[1:])



#Los transforma a un diccionario(?) lo que lee del documento cfg_args
#args = get_combined_args(parser)

args.model_path = "./Modelo"
args.source_path = "./train"


#Crea una clase solo con los argumentos de args
pipeline = pipeline.extract(args)
#Hace el diccionario, extiende de paramGroup
dataset = model.extract(args)

#Se crea el modelo de gaussianas
gaussians = GaussianModel(dataset.sh_degree)


#Background se crea como un tensor de 1x3
bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

#Crea la escena con las gaussianas
scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
views = scene.getTrainCameras()


# Check for folder
# Directory where the image will be saved
directory = "imgs"

# Check if the directory exists, and if not, create it
if not os.path.exists(directory):
    os.makedirs(directory)


for idx, view in enumerate(views):
    rendering = render(view, gaussians, pipeline, background)
    picture = rendering["render"]
    depths = rendering["depths"]

    torchvision.utils.save_image(picture,f"imgs/image{idx}.png")
    torchvision.utils.save_image(depths, f"imgs/depths{idx}.png")

#old = views[8]
#view = MiniCam(old.image_width, old.image_height, old.FoVy, old.FoVx, old.znear, old.zfar, old.world_view_transform, old.full_proj_transform) 

print("Done!")





