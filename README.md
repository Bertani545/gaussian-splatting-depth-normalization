# How to use the renderer
In the state it currently is, it can render a camera that was used to train the model. To do this, we have to define in **render.py**  at lines  36 and 37 the paths where the final model and the training data are located. Later, in line 58, the variable **views** contains all the cameras' parameters used in the training process. By specifying a certain camera number between brackets we can render different images.

Once done, to render the images we need to initiate a cuda enviroment. In the root directory run in a terminal
```shell
conda env create --file environment.yml
conda activate gaussian_splatting
```
and then run **render.py**

As it is now, the renderer will generate two images that reflects the depth of the scene. This will be changed so it generates a picture with depths and another one with the colors of the scene.

## Special Note
In order to use only the renderer in this state, the line 98 of ./arguments/\_\_init\_\_.py was changed to find the cfg gile at "./Modelo/cfg_args" so it's spected that such directory exists in the root directory. If a different directory is used, this line must be changed.


# How the rasterizer works

As you can see, in render.py the function **render** is called at line 58. This function is defined in **./gaussian_renderer/\_\_init\_\_.py** at line 18. There, after setting up the variables, the function creates a **GaussianRasterizer** called *rasterizer* and calls it with such variables. This call returns a Tuple with the rendered image, depths and radii.

The class **GaussianRasterizer** is defined in the the module **diff_gaussian_rasterization** wich is defined in **./submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/\_\_init\_\_.py** at line 171. When we "call it" in the previous step, we are actually calling the module **forward** which returns what the function **rasterize_gaussians** returns (This function is defined at line 21), which returns what **\_RasterizeGaussians.forward()** returns (line 44). This function then calls **\_C.rasterize_gaussians()** (line 86 or 92) which we can see is linked to the function **RasterizeGaussiansCUDA** in the file **./submodules/diff-gaussian-rasterization/ext.cpp**.  *RasterizeGaussiansCUDA* returns a C++ tuple and the linking allows it to tranform into a Python tuple.

*RasterizeGaussiansCUDA* is defined in **./submodules/diff-gaussian-rasterization/rasterize_points.cu** (line 35, there's also a .h file where the functions are defined). It is in this section where we have to allocate memory for new variables we would like to use. This function calls **CudaRasterizer::Rasterizer::forward** and then returns a Tuple of many torch objects which were modified by  *CudaRasterizer::Rasterizer::forward* to be handled in the previous function.

*CudaRasterizer::Rasterizer::forward* is defined in the file **./submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu**, line 198. Here, after culling, it preprocess the gaussians with **FORWARD::preprocess** and calls **FORWARD::render** to render the images we were looking. Both of this functions are defined in  **./submodules/diff-gaussian-rasterization/cuda_rasterizer/forward[.cu, .h]**. In such file, both functions exists to call the real CUDA function that will do all the job (**preprocessCUDA** and **renderCUDA**). Those are void functions so the results are saved directly to the memory we reserved before and kept track of it by pointers.

The code of **renderCUDA** is the one we have to change to actually get different results in our pictures. Of course, if we want to pass something new to the function, we have to modify everything at least starting at *rasterize_points.cu*.
