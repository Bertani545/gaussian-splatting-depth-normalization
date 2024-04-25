from random import randint, sample


class MyParams():
    def __init__(self, sceneIndices = None, n_cameras = -1, test = False, trainIndices = None, percentage = 1.0):
    	self.SceneIndices = sceneIndices.sort() if sceneIndices else None
        self.N_cameras = n_cameras

        self.MakeTest = test
        self.TrainIndices = trainIndices.sort() if trainIndices else None # Relative to the SceneIndices
        self.Percentage = max(0.0, min(percentage, 1.0))

class cameras_Subset :

    def __init__(self, scene : Scene, params = None : MyParams):
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
        self.TestSubset = []
        self.TrainSubset = []
        self.TestIndices = []
        self.TrainIndices = []

        if params.MakeTest:
        	if params.TrainIndices:
        		for idx in params.TrainIndices:
        			self.TrainSubset.append(self.CameraSubset[idx])
        			self.TrainIndices.append(params.SceneIndices[idx])

        	else:
        		trainSamples = int(len(self.CameraSubset) * percentage)

        		self.TrainSubset = random.sample(CameraSubset, trainSamples)
        		self.TrainIndices.append(params.SceneIndices[idx])

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

    def saveCameras(self):
    	with open("usedCameras.txt", 'w') as file:
            file.write("Train Indices:\n")
            for index in self.TrainIndices:
                file.write(str(index) + '\n')

            file.write("\nTest Indices:\n")
            for index in self.TestIndices:
                file.write(str(index) + '\n')

    def read_indices(self):
        with open("usedCameras.txt", 'r') as file:
            current_section = None
            for line in file:
                line = line.strip()
                if line == "Train Indices:":
                    current_section = "train"
                elif line == "Test Indices:":
                    current_section = "test"
                elif current_section == "train":
                    self.train_indices.append(int(line))
                elif current_section == "test":
                    self.test_indices.append(int(line))


