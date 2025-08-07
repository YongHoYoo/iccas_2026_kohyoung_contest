import os
import pandas as pd
from utils.generator import Generator
from utils.checker import Checker 

if __name__ == '__main__': 

    root_folder = 'manual_data/test' 

    # data generation
    g = Generator() 

    g.set_size_info( 
        pcb_size={ 
            'width': 30, 
            'height': 20
        },
        fov_size=8
    )
    g.set_parameter(capture_time=0.5, recon_time=0.5, max_core=8) 

    # add component manually
    # name, tl_x, tl_y, br_x, br_y, types, center, time s
    g.add_component(3, 3, 5, 5, 1, 1) # start point
    g.add_component(25, 15, 27, 17, 0, 1) 
    g.add_component(4, 14, 14, 16, 0, 5) 
    g.add_component(20, 9, 22, 14, 0, 4) 
    g.add_component(20, 5, 22, 7, 0, 2) 
    g.add_component(23, 5, 25, 7, 0, 2) 

    g.save_job_info(root_folder) 

    # make arbitrary output
    fov_df = pd.DataFrame(columns=['x', 'y', 'comp_idx']) 
    fov_df.loc[len(fov_df)] = [4, 4, "[0]"]
    fov_df.loc[len(fov_df)] = [26, 16, "[1]"]
    fov_df.loc[len(fov_df)] = [21, 12, "[3]"]
    fov_df.loc[len(fov_df)] = [22, 8, "[4, 5]"]
    fov_df.loc[len(fov_df)] = [12, 15, "[2]"]
    fov_df.loc[len(fov_df)] = [6, 15, "[2]"]

    fov_df.to_csv(os.path.join(root_folder, 'fov.csv')) 

    # result evaluation
    checker = Checker(root_folder) 
    ret = checker.cover_all_components() 
    checker.fov_inspection_order() 

    checker.save_pcb_view() 
    checker.save_timelog() 

