import os
import pandas as pd
from utils.generator import Generator
from utils.checker import Checker 

if __name__ == '__main__': 

    # data generation
    g = Generator() 

    g.set_size_info( 
        pcb_size={ 
            'length': 30, 
            'width': 20
        },
        fov_size={
            'pixel_l': 800, 
            'pixel_w': 800, 
            'scale_l': 10, 
            'scale_w': 10, 
            'margin': 1
        }
    )

    g.set_parameter(way=4, 
                    channel=4, 
                    fps=30, 
                    v_x=100, 
                    v_y=100, 
                    a_x=3000, 
                    a_y=3000)

    # add component manually
    # name, tl_x, tl_y, br_x, br_y, types, center, side, board, offset_x, offset_y, time s
    g.add_component('FID1', 3, 3, 5, 5, 2, 1, 0, 0, 0, 0, 10) 
    g.add_component('FID2', 25, 15, 27, 17, 2, 1, 0, 0, 0, 0, 10) 
    g.add_component('COMP1', 4, 14, 14, 16, 0, 0, 0, 0, 0, 0, 50) 
    g.add_component('COMP2', 20, 9, 22, 14, 0, 0, 0, 0, 0, 0, 40) 
    g.add_component('COMP3', 20, 5, 22, 7, 0, 0, 0, 0, 0, 0, 20) 
    g.add_component('COMP4', 23, 5, 25, 7, 0, 0, 0, 0, 0, 0, 20) 

    g.save_job_info('manual') 

    # make arbitrary output
    fov_df = pd.DataFrame(columns=['x', 'y', 'comp_idx']) 
    fov_df.loc[len(fov_df)] = [4, 4, "[0]"]
    fov_df.loc[len(fov_df)] = [26, 16, "[1]"]
    fov_df.loc[len(fov_df)] = [21, 12, "[3]"]
    fov_df.loc[len(fov_df)] = [22, 8, "[4, 5]"]
    fov_df.loc[len(fov_df)] = [12, 15, "[2]"]
    fov_df.loc[len(fov_df)] = [6, 15, "[2]"]

    fov_df.to_csv(os.path.join('manual', 'fov.csv')) 

    # result evaluation
    checker = Checker('manual') 
    ret = checker.cover_all_components() 
    checker.fov_inspection_order() 

    checker.save_pcb_view() 
    checker.save_timelog() 

