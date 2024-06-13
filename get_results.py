import os
import json
import numpy as np

def compute_means_and_generate_tables(base_path):
    with open('sources.txt', 'r') as file:
        sources = file.readlines()
    scenes = ["results/" + source.strip().split(os.path.sep)[-1] for source in sources]    

    cameras = ["3_cams", "4_cams", "5_cams", "8_cams", "10_cams"]
    depth_types = ["depth", "no_depth"]
    subfolders = ["0", "1", "2", "3", "4"]
    
    results = {}
    
    for scene in scenes:
        results[scene] = {}
        for cam in cameras:
            results[scene][cam] = { "depth": {"SSIM": [], "PSNR": [], "LPIPS": []},
                                    "no_depth": {"SSIM": [], "PSNR": [], "LPIPS": []} }
            for depth_type in depth_types:
                for subfolder in subfolders:
                    result_file = os.path.join(base_path, scene, cam, depth_type, subfolder, "results.json")
                    if os.path.exists(result_file):
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                            results[scene][cam][depth_type]["SSIM"].append(data["ours_30000"]["SSIM"])
                            results[scene][cam][depth_type]["PSNR"].append(data["ours_30000"]["PSNR"])
                            results[scene][cam][depth_type]["LPIPS"].append(data["ours_30000"]["LPIPS"])
    
    for scene in scenes:
        print(generate_latex_table(scene, results[scene]))

def generate_latex_table(scene, data):
    
    ssim_depth = []
    psnr_depth = []
    lpips_depth = []
    ssim_no_depth = []
    psnr_no_depth = []
    lpips_no_depth = []
    
    for cam in ["3_cams", "4_cams", "5_cams", "8_cams", "10_cams"]:
        ssim_depth.append(np.mean(data[cam]["depth"]["SSIM"]))
        psnr_depth.append(np.mean(data[cam]["depth"]["PSNR"]))
        lpips_depth.append(np.mean(data[cam]["depth"]["LPIPS"]))
        ssim_no_depth.append(np.mean(data[cam]["no_depth"]["SSIM"]))
        psnr_no_depth.append(np.mean(data[cam]["no_depth"]["PSNR"]))
        lpips_no_depth.append(np.mean(data[cam]["no_depth"]["LPIPS"]))


    return r"""
    \begin{table}[H]
        \centering
        \begin{tabular}{|c|c|c|c|c|c|}
            \hline
            Camaras &3&4&5&8&10\\\hline
            SSIM Profundidad $\uparrow$ & """+ ' & '.join(f"{v:.5f}" for v in ssim_depth) + r""" \\
            SSIM Sin Profundidad $\uparrow$ & """+ ' & '.join(f"{v:.5f}" for v in ssim_no_depth) + r""" \\\hline
            PSNR Profundidad $\uparrow$ & """+ ' & '.join(f"{v:.5f}" for v in psnr_depth) + r""" \\
            PSNR Sin Profundidad $\uparrow$ & """+ ' & '.join(f"{v:.5f}" for v in psnr_no_depth) + r""" \\\hline
            LPIPS Profundidad $\downarrow$ & """+ ' & '.join(f"{v:.5f}" for v in lpips_depth) + r""" \\
            LPIPS Sin Profundidad $\downarrow$ & """+ ' & '.join(f"{v:.5f}" for v in lpips_no_depth) + r""" \\\hline
        \end{tabular}
        \caption{{Errores por número de cámaras para la escena """+ scene + r"""}}
        \label{{tab:error_cameras_"""+ scene + r"""}}
    \end{table}
    """

if __name__ == "__main__":
    base_path = "./"
    compute_means_and_generate_tables(base_path)

