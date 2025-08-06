import os
import csv
import random
import string 
import numpy as np
from rtree import index

def random_string(length=6):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

class Generator(): 
    def __init__(self): 
        self.components = [] 

    def initialize(self, pcb_size, n_components, component_info, big_component_info): 
        self.pcb_length = pcb_size['length'] 
        self.pcb_width = pcb_size['width'] 
        self.n_components = n_components 
        self.component_info = component_info 
        self.big_component_info = big_component_info 
        self.components = [] 

    def biased_size(self, min_size, max_size, power=2.0):
        r = random.random() ** power
        return round(min_size + (max_size - min_size) * r, 2)

    def set_parameter(self, capture_time, recon_time, max_thread=None, max_grab=None): 
        self.parameter = { 
            'capture_time': capture_time, 
            'recon_time': recon_time, 
            'max_thread': random.randint(8, 16) if max_thread is None else max_thread, 
            'max_grab': random.randint(3, 8) if max_grab is None else max_grab, 
            'v_x': 1000, 
            'v_y': 1000, 
            'a_x': 9800, 
            'a_y': 9800 
        }

    def set_size_info(self, pcb_size, fov_size): 
        self.pcb_size = pcb_size 
        self.fov_size = fov_size 

    def generate_fiducials(self, max_offset=10): 
        # Assume fiducial size is 5 by 5 
        size = 5
        diagonal_points = ["top-left", "bottom-right", "top-right", "bottom-left"]
        corner = random.choice(diagonal_points)

        corner_base = {
            "top-left": (max_offset, max_offset),
            "top-right": (self.pcb_length - size - max_offset, max_offset),
            "bottom-left": (max_offset, self.pcb_width - size - max_offset),
            "bottom-right": (self.pcb_length - size - max_offset, self.pcb_width - size - max_offset),
        }

        fiducials = []
        while True: 
            base_x, base_y = corner_base[corner]
            offset_x = random.randint(-max_offset, max_offset)
            offset_y = random.randint(-max_offset, max_offset)
            tl_x = min(max(base_x + offset_x, 0), self.pcb_length - size)
            tl_y = min(max(base_y + offset_y, 0), self.pcb_width - size)
            br_x = tl_x + size
            br_y = tl_y + size

            overlap = False 
            for comp in self.components: 
                if not (br_x <= comp[1] or tl_x >= comp[3] or br_y <= comp[2] or tl_y >= comp[4]): 
                    overlap = True 
                    break 
        
            if not overlap: 
                fiducials.append([tl_x, tl_y, br_x, br_y, 2, size*size/100.])
                break 


        self.components = fiducials + self.components 

    def generate_components(self, grid_resolution=10.0, cluster_count=[10,30], n_component_per_cluster=[20,30], cluster_radius=[10.,50.], bernoulli_prob=0.3): 

        components = [] 
        grid_x = int(self.pcb_length / grid_resolution) 
        grid_y = int(self.pcb_width / grid_resolution) 

        # Step 1. Normal grid-based components 
        for gx in range(grid_x): 
            for gy in range(grid_y): 
                if random.random() > bernoulli_prob: 
                    continue

                tl_x = gx * grid_resolution + random.uniform(0.05, 0.95)
                tl_y = gy * grid_resolution + random.uniform(0.05, 0.95) 

                if random.random() < self.big_component_info['ratio']: 
                    min_size, max_size = self.big_component_info['size'] 
                    l = self.biased_size(min_size, max_size)
                    w = self.biased_size(min_size, max_size)

                    r = random.random() 
                    if r < 0.25: 
                        l *= 0.5 
                    elif r < 0.5: 
                        w *= 0.5 

                else: 
                    min_size, max_size = self.component_info['size'] 
                    l = self.biased_size(min_size, max_size)
                    w = self.biased_size(min_size, max_size)

                br_x = tl_x + l
                br_y = tl_y + w

                if br_x > self.pcb_length or br_y > self.pcb_width: 
                    continue 

                components.append([tl_x, tl_y, br_x, br_y, 0, l*w/100.])

        # Step 2. Generate clusters near some of the rectangles 
        clustered_components = components.copy() 
        random.shuffle(clustered_components) 

        for base in clustered_components[:random.randint(cluster_count[0], cluster_count[1])]: 

            base_tl_x, base_tl_y, base_br_x, base_br_y, types, time = base 
            center_x = (base_tl_x + base_br_x) / 2.0 
            center_y = (base_tl_y + base_br_y) / 2.0 

            for _ in range(random.randint(n_component_per_cluster[0], n_component_per_cluster[1])): 
                radius = random.uniform(cluster_radius[0], cluster_radius[1])
                offset_x = random.uniform(-radius, radius)
                offset_y = random.uniform(-radius, radius)

                tl_x = min(max(center_x + offset_x, 0), self.pcb_length - max_size)
                tl_y = min(max(center_y + offset_y, 0), self.pcb_width - max_size)

                min_size, max_size = self.component_info['size'] 
                l = self.biased_size(min_size, max_size)
                w = self.biased_size(min_size, max_size)
                br_x = tl_x + l
                br_y = tl_y + w

                if br_x > self.pcb_length or br_y > self.pcb_width: 
                    continue 

                components.append([tl_x, tl_y, br_x, br_y, types, l*w/100.]) 

        # Step 3. Remove overlaps and limit total count 
        final_components = [] 
        random.shuffle(components) 
        idx = index.Index() 

        for i, c in enumerate(components): 
            tl_x, tl_y, br_x, br_y = c[0], c[1], c[2], c[3]
            hits = list(idx.intersection((tl_x, tl_y, br_x, br_y))) 
            if not hits: 
                final_components.append(c) 
                idx.insert(i, (tl_x, tl_y, br_x, br_y)) 

        self.components = final_components[:self.n_components] 

    def add_component(self, tl_x, tl_y, br_x, br_y, types, time): 
        self.components.append([tl_x, tl_y, br_x, br_y, types, time])

    def save_job_info(self, folder): 

        os.makedirs(folder, exist_ok=True) 
        # component info
        component_file = folder + '/component.csv' 
        with open(component_file, mode='w', newline='') as f: 
            writer = csv.writer(f) 
            writer.writerow(["", "tl_x", "tl_y", "br_x", "br_y", "type", "time"])
            
            for idx, row in enumerate(self.components): 
                writer.writerow([idx, 
                                f"{row[0]:.2f}", #tl_x
                                f"{row[1]:.2f}", #tl_y
                                f"{row[2]:.2f}", #br_x
                                f"{row[3]:.2f}", #br_y
                                row[4], # type
                                row[5]
                                ] 
                            ) 

        # size info 
        size_file = folder + '/size.csv' 
        with open(size_file, mode='w', newline='') as f: 
            writer = csv.writer(f) 
            writer.writerow(["axis", "pcb_size", "fov_size"]) 
            writer.writerow(["x", self.pcb_size['length'], self.fov_size]) 
            writer.writerow(["y", self.pcb_size['width'], self.fov_size])
        
        # parameter info
        param_file = folder + '/parameter.csv' 
        with open(param_file, mode='w', newline='') as f: 
            writer = csv.writer(f) 
            writer.writerow(["capture_time", "recon_time", "max_thread", "max_grab", "v_x", "v_y", "a_x", "a_y"]) 
            writer.writerow([self.parameter['capture_time'], self.parameter['recon_time'], self.parameter['max_thread'], self.parameter['max_grab'], self.parameter['v_x'], self.parameter['v_y'], self.parameter['a_x'], self.parameter['a_y']])
