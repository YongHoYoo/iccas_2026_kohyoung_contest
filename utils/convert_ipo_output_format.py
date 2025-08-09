import os
import numpy as np
import pandas as pd

if __name__ == '__main__': 

    csv_folder = 'csv' 
    root_folder = 'simulation_data' 

    for sub_folder in os.listdir(csv_folder): 

        component_file = os.path.join(csv_folder, sub_folder, 'component.csv') 
        fov_file = os.path.join(csv_folder, sub_folder, 'output_fov_ipo_1.csv') 
        size_file = os.path.join(csv_folder, sub_folder, 'size.csv') 

        fov_df = pd.read_csv(fov_file, index_col=0) 
        fov_df.drop(['sub_type', 'step', 'big', 'array', 'z', 'array_idx'], axis=1, inplace=True)
        fov_df['type'] = fov_df['type'].replace({7: 1})
        fov_df.to_csv(os.path.join(root_folder, sub_folder, 'output_fov.csv')) 

