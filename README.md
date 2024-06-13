# How to use the program

We need to initiate a cuda enviroment. In the root directory run in a terminal
```shell
conda env create --file environment.yml
conda activate gaussian_splatting
```

Once that done, we wirte the route to all the models we want to test in *sources.txt*. The main program that runs the experiment is **experiment.py**. Inside it you can find commands that will be run in the terminal. If you wish to change the value of $\lambda_{TV}$ you would have to change the value next to --TVL in the first command. To run it just use

```shell
python experiment.py
```

At 30000 iterations, each scene takes around 8 hours to complete. You can also change the iterations in the command if desired.

Once the experiment is done, the output will be stored in a folder named *results*. To get the $\LaTeX$ tables to report the results as in the report, simply run
```shell
python get_results.py
```
and the tables will be ready to copy and paste in the terminal.

Finally, if you want to render the depth of certain obtained model and camera you can use *render\_single\_image.py*. To use it run:
```shell
python render_single_image.py -m path_to_3DGS_output --camera camera_number
```
A 3DGS output is the folder that contains a file named *cfg_args*. In our experiment, the path to the outputs is *results/scene_name/n_cameras/{depth/no_depth}/{0-5}.

(The experiment for lambda is also included but is just a modification of the main file)

# How the rasterizer works if you ever need to modify it (Bakcward pass not included)

As you can see, in render.py the function **render** is called at line 58. This function is defined in **./gaussian_renderer/\_\_init\_\_.py** at line 18. There, after setting up the variables, the function creates a **GaussianRasterizer** called *rasterizer* and calls it with such variables. This call returns a Tuple with the rendered image, depths and radii.

The class **GaussianRasterizer** is defined in the the module **diff_gaussian_rasterization** wich is defined in **./submodules/diff-gaussian-rasterization/diff_gaussian_rasterization/\_\_init\_\_.py** at line 171. When we "call it" in the previous step, we are actually calling the module **forward** which returns what the function **rasterize_gaussians** returns (This function is defined at line 21), which returns what **\_RasterizeGaussians.forward()** returns (line 44). This function then calls **\_C.rasterize_gaussians()** (line 86 or 92) which we can see is linked to the function **RasterizeGaussiansCUDA** in the file **./submodules/diff-gaussian-rasterization/ext.cpp**.  *RasterizeGaussiansCUDA* returns a C++ tuple and the linking allows it to tranform into a Python tuple.

*RasterizeGaussiansCUDA* is defined in **./submodules/diff-gaussian-rasterization/rasterize_points.cu** (line 35, there's also a .h file where the functions are defined). It is in this section where we have to allocate memory for new variables we would like to use. This function calls **CudaRasterizer::Rasterizer::forward** and then returns a Tuple of many torch objects which were modified by  *CudaRasterizer::Rasterizer::forward* to be handled in the previous function.

*CudaRasterizer::Rasterizer::forward* is defined in the file **./submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.cu**, line 198. Here, after culling, it preprocess the gaussians with **FORWARD::preprocess** and calls **FORWARD::render** to render the images we were looking. Both of this functions are defined in  **./submodules/diff-gaussian-rasterization/cuda_rasterizer/forward[.cu, .h]**. In such file, both functions exists to call the real CUDA function that will do all the job (**preprocessCUDA** and **renderCUDA**). Those are void functions so the results are saved directly to the memory we reserved before and kept track of it by pointers.

The code of **renderCUDA** is the one we have to change to actually get different results in our pictures. Of course, if we want to pass something new to the function, we have to modify everything at least starting at *rasterize_points.cu*.
