import random
import string
import numpy as np
import os
from utils.generator import Generator

if __name__ == "__main__":

    root_folder = 'simulation_data' 
    fov_size = [40, 50, 60] # um

    for pcb_idx in range(10): 

        pcb_size = dict() 
        pcb_size['width'] = random.randint(100, 500) # width: y direction
        pcb_size['height'] = pcb_size['width'] * random.uniform(0.6, 1.0) # height: x direction
        n_components = int(pcb_size['width']) * random.randint(80, 100) 

        if random.random() < 0.5: 
            pcb_size['height'], pcb_size['width'] = pcb_size['width'], pcb_size['height'] 

        big_component_ratio = random.uniform(0.03, 0.05) 
        big_component_info = {'size': [40, 60], 'ratio': big_component_ratio} 
        component_info = {'size': [1, 15], 'ratio': 1-big_component_ratio}

        g = Generator() 
        g.initialize(pcb_size, n_components, component_info, big_component_info)
        g.generate_components()
        g.generate_fiducials() 

        for fov in fov_size: 
            for core in [8, 16]: 
                g.set_size_info(pcb_size, fov)
                g.set_parameter(capture_time=fov/100, recon_time=fov/100, max_core=core) 
                folder = f"{root_folder}/PCB{pcb_idx:04d}_{fov}um_thread{core}" 
                g.save_job_info(folder) 
