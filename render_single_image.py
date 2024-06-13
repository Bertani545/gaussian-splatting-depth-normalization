#usage: python render_single_image.py -m modelOuputPath -camera cameraNumber(ID)
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

from custom_classes import *

from PIL import Image


def apply_basic_colormap(depth_np):

    # Create an empty array for the RGB image
    depth_colored = np.zeros((depth_np.shape[0], depth_np.shape[1], 3), dtype=np.uint8)

    # Apply a simple colormap (blue to red gradient)
    depth_colored[..., 0] = (255 * depth_np).astype(np.uint8)  # Red channel
    depth_colored[..., 2] = (255 * (1 - depth_np)).astype(np.uint8)  # Blue channel

    # Convert to PIL image
    depth_colored_pil = Image.fromarray(depth_colored)
    
    return depth_colored_pil


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
parser.add_argument("--camera", type=int, default=0)
args = parser.parse_args(sys.argv[1:])



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
scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

#Indices
usedCameras = cameras_Subset()
usedCameras.read_indices(dataset.model_path, scene)

viewsTrain = usedCameras.getTrainSubset()
viewsTest  = usedCameras.getTestSubset()

if args.camera in usedCameras.TrainIndices:
    view = viewsTrain[usedCameras.TestIndices.index(args.camera)]
elif args.camera in usedCameras.TestIndices:
    view = viewsTest[usedCameras.TestIndices.index(args.camera)]
else:
    print("No camera found")

#Renderiza la imagen
rendering = render(view, gaussians, pipeline, background)

picture = rendering["render"]
depths = rendering["depths"]

#Transform deptht to a colored map
depths = depths - depths.min()
depths = depths/depths.max()
depths_np = depths.squeeze().cpu().detach().numpy()

depths_rgb = apply_basic_colormap(depths_np)

torchvision.utils.save_image(picture, "image.png")
#torchvision.utils.save_image(depths_rgb, "depths.png")
depths_rgb.save("depths.png")

print("Done!")

