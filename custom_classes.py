import os
from random import randint, sample
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import torch
from torch import nn

from utils.graphics_utils import BasicPointCloud


def rodrigues_Matrix(axis, c, s):
    #c = np.cos(angle)
    #s = np.sin(angle)
    oc = 1 - c

    x = axis[0]
    y = axis[1]
    z = axis[2]

    mat = np.zeros((3,3))

    mat[0][0] = oc * x * x + c;
    mat[0][1] = oc * x * y - z * s;
    mat[0][2] = oc * z * x + y * s;

    mat[1][0] = oc * x * y + z * s;
    mat[1][1] = oc * y * y + c;
    mat[1][2] = oc * y * z - x * s;

    mat[2][0] = oc * z * x - y * s;
    mat[2][1] = oc * y * z + x * s;
    mat[2][2] = oc * z * z + c;

    return mat


class NewCameras():
    def __init__(self, scene, res = 1.0):
        self.pointsCenter = torch.mean(scene.gaussians._xyz, dim=0).cpu().numpy()

        positions = np.array([camera.T for camera in scene.train_cameras[res]])
        positions_reshaped = positions.reshape(-1, 1, 3)
        self.distances = np.linalg.norm(positions_reshaped - positions, axis=2)
        self.sorted_indices = np.argsort(self.distances, axis=1)

        self.cameras = scene.train_cameras[res]  #Hopefully is the reference


    def __call__(self):
        camera_id = randint(0, len(self.cameras)-1)
        new_camera = MiniCam_FromCam(self.cameras[camera_id])

        #Take closest one
        cameras_idx = np.array([camera_id, self.sorted_indices[camera_id,1]])

        #Transform new camera position and rotation
        radius = np.linalg.norm(self.pointsCenter - self.cameras[camera_id].T)

        random_direction = np.random.uniform(low=-1, high=1, size=3)
        random_direction = random_direction / np.linalg.norm(random_direction)

        new_camera.T = self.cameras[camera_id].T + random_direction * self.distances[cameras_idx[0], cameras_idx[1]] / 1.5
        push = self.pointsCenter - new_camera.T
        new_camera.T += (np.linalg.norm(push) - radius) * (push/np.linalg.norm(push))

        #Rotation
        front = np.array([0,0,1.0]) #Espero
        direction = - (0.2 * random_direction - self.cameras[camera_id].R @ front)
        direction /= np.linalg.norm(direction)

        axis = np.cross(front, direction)
        new_camera.R = rodrigues_Matrix(axis/np.linalg.norm(axis), np.dot(front, direction), np.linalg.norm(axis))
        
        new_camera.recalculate()
        return new_camera

class MiniCam_FromCam:
    def __init__(self, cam, scale=1.0):
        self.image_width = cam.image_width
        self.image_height = cam.image_height    
        self.FoVy = cam.FoVy
        self.FoVx = cam.FoVx
        self.znear = cam.znear
        self.zfar = cam.zfar

        self.R = cam.R
        self.T = cam.T
        self.trans = cam.trans
        self.scale = cam.scale

        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def recalculate(self):
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MyParams():
    def __init__(self, sceneIndices = None, n_cameras = -1, test = False, trainIndices = None, percentage = 1.0, allPoints = False, useDepths = True, lbd = 1.0):
        self.SceneIndices = sceneIndices.sort() if sceneIndices else None
        self.N_cameras = n_cameras

        self.MakeTest = test
        self.Percentage = max(0.0, min(percentage, 1.0))
        self.AllPoints = allPoints

        self.UseDepths = useDepths
        self.L_TVL = lbd

        #Modified when starting an scene
        self.TrainIndices = trainIndices.sort() if trainIndices else None # Relative to the SceneIndices
        self.TestIndices = []

class cameras_Subset :

    def __init__(self):
        self.AllCameras = []
        self.SceneIndices = []
        self.TestIndices = []
        self.TrainIndices = []
        self.TestSubset = []
        self.TrainSubset = []

    def getSubset(self):
        return self.CameraSubset

    def getTrainSubset(self):
    	return self.TrainSubset

    def getTestSubset(self):
    	return self.TestSubset

    def loadViews(self):
        if len(self.TrainIndices) > 0:
            for idx in self.TrainIndices:
                self.TrainSubset.append(self.AllCameras[idx])

        if len(self.TestIndices) > 0:
            for idx in self.TestIndices:
                self.TestSubset.append(self.AllCameras[idx])

    def saveCameras(self, outputPath):
    	path = os.path.join(outputPath, "usedCameras.txt")
    	with open(path, 'w') as file:
            file.write("Train Indices:\n")
            for index in self.TrainIndices:
                file.write(str(index) + '\n')

            file.write("\nTest Indices:\n")
            for index in self.TestIndices:
                file.write(str(index) + '\n')

    def read_indices(self, outputPath, scene):
        self.AllCameras = scene.getTrainCameras().copy()

        path = os.path.join(outputPath, "usedCameras.txt")
        with open(path, 'r') as file:
            current_section = None
            for line in file:
                line = line.strip()
                if line == "Train Indices:":
                    current_section = "train"
                elif line == "Test Indices:":
                    current_section = "test"
                elif line == "":
                    pass
                elif current_section == "train":
                    self.TrainIndices.append(int(line))
                elif current_section == "test":
                    self.TestIndices.append(int(line))

        self.loadViews()


