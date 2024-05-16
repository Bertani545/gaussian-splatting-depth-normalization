import os
from random import randint, sample
#from scene import Scene
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import torch
from torch import nn

#from scene.dataset_readers import SceneInfo
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

"""
Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  points3D_ids=cam_info.point3D_ids)
"""


class NewCameras():
    def __init__(self, scene, res = 1.0):
        self.pointsCenter = torch.mean(scene.gaussians._xyz, dim=0).cpu().numpy()

        positions = np.array([camera.T for camera in scene.train_cameras[res]])
        positions_reshaped = positions.reshape(-1, 1, 3)
        self.distances = np.linalg.norm(positions_reshaped - positions, axis=2)
        self.sorted_indices = np.argsort(self.distances, axis=1)

        self.cameras = scene.train_cameras[res]  #Hopefully is the reference :b


    def __call__(self):
        camera_id = randint(0, len(self.cameras)-1)
        new_camera = MiniCam_FromCam(self.cameras[camera_id])

        #Take closest one
        cameras_idx = np.array([camera_id, self.sorted_indices[camera_id,1]])

        #Transform new camera position and rotation
        radius = np.linalg.norm(self.pointsCenter - self.cameras[camera_id].T)

        random_direction = np.random.normal(size=3)
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
    """
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
        """

class MyParams():
    def __init__(self, sceneIndices = None, n_cameras = -1, test = False, trainIndices = None, percentage = 1.0, allPoints = False, useDepths = True):
        self.SceneIndices = sceneIndices.sort() if sceneIndices else None
        self.N_cameras = n_cameras

        self.MakeTest = test
        self.Percentage = max(0.0, min(percentage, 1.0))
        self.AllPoints = allPoints

        self.UseDepths = useDepths

        #Modified when starting an scene
        self.TrainIndices = trainIndices.sort() if trainIndices else None # Relative to the SceneIndices
        self.TestIndices = []

class cameras_Subset :

    #def __init__(self, scene : Scene = None, params : MyParams = None):
    def __init__(self):
        self.AllCameras = []
        self.SceneIndices = []
        self.TestIndices = []
        self.TrainIndices = []
        self.TestSubset = []
        self.TrainSubset = []


        '''
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
        '''

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


