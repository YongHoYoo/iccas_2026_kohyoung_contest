import random
import string
import numpy as np
import os
from utils.generator import Generator

if __name__ == "__main__":

    root_folder = 'simulation_data' 
    fov_size = [40, 50, 60] 

    for pcb_idx in range(10): #144 * 10

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
        g.generate_fiducials() 

        for fov in fov_size: 
            g.set_size_info(pcb_size, fov) 
            g.set_parameter(capture_time=fov/100, recon_time=fov/100) 
            folder = f"{root_folder}/PCB{pcb_idx:04d}_{fov}um" 
            g.save_job_info(folder) 
