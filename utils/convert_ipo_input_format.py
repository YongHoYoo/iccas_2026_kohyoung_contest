import os
import csv
import pandas as pd

if __name__ == '__main__': 

    root_folder = 'simulation_data' 

    os.makedirs('csv', exist_ok=True) 

    for folder in os.listdir(root_folder): 
            f = os.path.join(root_folder, folder) 

            csv_folder = os.path.join('csv', folder)
            os.makedirs(csv_folder, exist_ok=True) 

            component_file = os.path.join(f, 'component.csv') 
            size_file = os.path.join(f, 'size.csv') 

            # convert component format 
            component_df = pd.read_csv(component_file)

            component_df['name'] = 'name' 
            component_df['step'] = 1
            component_df['array'] = 0 
            component_df['array_group'] = 1
            component_df['type'] = component_df['type'].replace({1: 4, 2: 7})
            component_df['sub_type'] = 0 
            component_df['center'] = 0 
            component_df['side'] = 0 
            component_df['enable'] = 1 
            component_df['board'] = 0 
            component_df['mfov'] = 0 
            component_df['offset_x'] = 0 
            component_df['offset_y'] = 0 
            component_df['z'] = -1 
            component_df['body_condi'] = 2047
            component_df['lead_condi'] = 2047 

            component_df = component_df[
                ['name', 'tl_x', 'tl_y', 'br_x', 'br_y', 'step', 'array', 'array_group', 'type', 'sub_type', 'center', 'side', 'enable', 'board', 'mfov', 'offset_x', 'offset_y', 'z', 'body_condi', 'lead_condi']
            ]

            component_df.to_csv(csv_folder + '/component.csv', encoding='utf-8') 

            # convert size format
            size_df = pd.read_csv(size_file) 
            size_df['margin_size'] = 0 
            size_df['side_fov_size'] = size_df['fov_size'] / 2 
            size_df['side_margin_size'] = 0 
            size_df =size_df.drop(columns=['axis'])


            size_df.to_csv(csv_folder + '/size.csv', encoding='utf-8') 

            # make array 
            array_file = csv_folder + '/array.csv' 
            with open(array_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["id", "x", "y", "angle", "group"])
                writer.writerow([0, 1, 0, 0, 1])
