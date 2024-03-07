from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

import torch
import torchvision

from gaussian_renderer import GaussianModel
from gaussian_renderer import render

from utils.general_utils import safe_state

from scene import Scene
import os
from os import makedirs


#Se crea el parser
parser = ArgumentParser(description="Testing script parameters")
#Argumentos del modelo
model = ModelParams(parser, sentinel=True)
#Se crea la clase pipeline y agregamos mas al parser
pipeline = PipelineParams(parser)


parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--skip_train", action="store_true")
parser.add_argument("--skip_test", action="store_true")
parser.add_argument("--quiet", action="store_true")



#Los transforma a un diccionario(?) lo que lee del documento cfg_args
args = get_combined_args(parser)


args.model_path = "./Modelo"
args.source_path = "./train"


#Crea una clase solo con los argumentos de args
pipeline = pipeline.extract(args)
#Hace una clase GroupParams con los datos de args
dataset = model.extract(args)


#Se crea el modelo de gaussianas
gaussians = GaussianModel(dataset.sh_degree)


#Background se crea como un tensor de 1x3 ????
bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


scene = Scene(dataset, gaussians, args.iteration, shuffle=False)
views = scene.getTrainCameras()

rendering = render(views[10], gaussians, pipeline, background)

picture = rendering["render"]
depths = rendering["depths"]


torchvision.utils.save_image(picture, "image.png")
torchvision.utils.save_image(depths, "depths.png")












