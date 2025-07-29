import random
import string
import numpy as np
import os
from utils.generator import Generator

if __name__ == "__main__":

    root_folder = 'simulation_data' 

    # the number of pixels
    fov_resolutions = {
        '4M' : [2048, 2000], 
        '8M': [2844, 2816], 
        '12M': [3072, 4096]
        '25M': [5120, 5120]
    }

    # um/pixel
    fov_scale = [10.01, 14.85, 19.99]

    # way
    way = [4, 8]

    # channel 
    channel = [4,8]

    # fps 
    fps = [30, 60, 90]

    for pcb_idx in range(100): #144 * 10

        pcb_size = dict() 
        pcb_size['length'] = random.randint(100, 500) 
        pcb_size['width'] = pcb_size['length'] * random.uniform(0.5, 0.8) 

        if random.random() < 0.5: 
            pcb_size['length'], pcb_size['width'] = pcb_size['width'], pcb_size['length'] 

        n_components = int(pcb_size['length']) * random.randint(50, 80) 
        big_component_ratio = random.uniform(0.01, 0.05) 

        big_component_info = {'size': [25, 50], 'ratio': big_component_ratio} 
        component_info = {'size': [1, 15], 'ratio': 1-big_component_ratio}

        g = Generator() 
        g.initialize(pcb_size, n_components, component_info, big_component_info)
        g.generate_components()
        g.assign_option() 
        g.generate_fiducials() 

        for k, v in fov_resolutions.items(): 
            for scale in fov_scale: 
                margin = random.uniform(1, 3)      # mm

                fov_size = { 
                    'pixel_l': v[0],               # pixel
                    'pixel_w': v[1],               # pixel
                    'scale_l': scale,              # um/pixel
                    'scale_w': scale,              # um/pixel
                    'margin': random.uniform(1, 3) # mm
                }

                for w in way: 
                    for c in channel: 
                        for f in fps:
                            g.set_size_info(pcb_size, fov_size) 
                            g.set_parameter(w, c, f) 
                            folder = f"{root_folder}/PCB{pcb_idx:04d}_{k}{int(scale)}_W{int(w)}_CH{int(c)}_FPS{f}"
                            g.save_job_info(folder) 
