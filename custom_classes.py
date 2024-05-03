import os
from random import randint, sample
from scene import Scene
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import torch
from torch import nn

from scene.dataset_readers import SceneInfo
from utils.graphics_utils import BasicPointCloud

def rodrigues_Matrix(axis, angle):
    c = np.cos(angle)
    s = np.sin(angle)
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

'''
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array
'''

def modifyPointCloud( sceneInfo : SceneInfo):
    sceneInfo.point_cloud
    


class MiniCam_FromCam:
    def __init__(self, cam, rot_offset=np.radians(0.1), trans_offset=np.array([0.01, 0.02, 0.03]), trans=np.array([0.0, 0.0, 0.0]), scale=1.0):
        self.image_width = cam.image_width
        self.image_height = cam.image_height    
        self.FoVy = cam.FoVy
        self.FoVx = cam.FoVx
        self.znear = cam.znear
        self.zfar = cam.zfar

        # Apply small rotation offset
        R_offset = np.eye(3)  # Identity matrix for no initial rotation offset
        if rot_offset > 0:
            # Choose an axis for rotation (e.g., x-axis)
            axis = np.random.rand(3)
            axis = axis / np.linalg.norm(axis)
            R_offset = rodrigues_Matrix(axis, rot_offset)

            # Apply small translation offset
            T_offset = trans_offset

        # Combine original and offset values
        R = cam.R @ R_offset
        T = cam.T + T_offset

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class MyParams():
    def __init__(self, sceneIndices = None, n_cameras = -1, test = False, trainIndices = None, percentage = 1.0, allPoins = False):
        self.SceneIndices = sceneIndices.sort() if sceneIndices else None
        self.N_cameras = n_cameras

        self.MakeTest = test
        self.TrainIndices = trainIndices.sort() if trainIndices else None # Relative to the SceneIndices
        self.Percentage = max(0.0, min(percentage, 1.0))
        self.AllPoints = allPoints

class cameras_Subset :

    def __init__(self, scene : Scene = None, params : MyParams = None):
        
        self.AllCameras = []
        self.SceneIndices = []
        self.TestIndices = []
        self.TrainIndices = []
        self.TestSubset = []
        self.TrainSubset = []

        ## ----- To be deleted -------
        if scene and params:
            self.AllCameras = scene.getTrainCameras().copy()
        

            if params.SceneIndices :
                self.SceneIndices = params.SceneIndices
            elif params.N_cameras  > 0:
                self.SceneIndices = [randint(0, len(self.AllCameras)) for _ in range(min(len(self.AllCameras)-1, params.N_cameras))]
            else:
                self.SceneIndices = [_ for _ in range(len(self.AllCameras))]
        
            self.CameraSubset = []
            for idx in self.SceneIndices:
                self.CameraSubset.append(self.AllCameras[idx])

            print(f"Number of cameras: {len(self.CameraSubset)}")
            print(f"Working with cameras {self.SceneIndices}")


        #Construct the train and test sets
       
            if params.MakeTest:
                if params.TrainIndices:
                    for idx in params.TrainIndices:
                        self.TrainSubset.append(self.CameraSubset[idx])
                        self.TrainIndices.append(params.SceneIndices[idx])

                else:
                    trainSamples = int(len(self.CameraSubset) * params.Percentage)


                    self.TrainIndices = sample(params.SceneIndices, trainSamples)
                    for idx in self.TrainIndices:
                        self.TrainSubset.append(self.AllCameras[idx])

                self.TestSubset = [_ for _ in self.CameraSubset if _ not in self.TrainSubset]
                self.TestIndices = [_ for _ in params.SceneIndices if _ not in self.TrainIndices]

            else:
                self.TrainSubset = self.CameraSubset.copy()
                self.TrainIndices = params.SceneIndices.copy()

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

    def read_indices(self, outputPath, scene : Scene):
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

