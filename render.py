from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args

import torch
import torchvision

from gaussian_renderer import render
from gaussian_renderer import GaussianModel



#Se crea el parser
parser = ArgumentParser(description="Testing script parameters")
#Argumentos del modelo
model = ModelParams(parser, sentinel=True)
#Se crea la clase pipeline y agregamos mas al parser
pipeline = PipelineParams(parser)
#Los transforma a un diccionario(?) lo que lee del documento cfg_args
args = get_combined_args(parser)
#Crea una clase solo con los argumentos de args
pipeline = pipeline.extract(args)
#Hace el diccionario, extiende de paramGroup
dataset = model.extract(args)

#Se crea el modelo de gaussianas
gaussians = GaussianModel(dataset.sh_degree)


#Background se crea como un tensor de 1x3 ????
bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

#Crea la escena con las gaussianas
scene = Scene(dataset, gaussians, args.iteration, shuffle=False)
views = scene.getTrainCameras()

#Renderiza la imagen
rendering = render(views[0], gaussians, pipeline, background)["render"]
torchvision.utils.save_image(rendering, "resultado.png")













