import os
import numpy as np
import csv
import pandas as pd
import plotly.graph_objects as go

def get_component_rect(df): 

    normal_df = df[df['type'] == 0]
    usermark_df = df[df['type'] == 4] 
    fiducial_df = df[df['type'] == 7] 
    side_df = df[df['side'] == 1] 
    board_df = df[df['board'] == 1] 

    data = dict() 
    data['normal'] = {
        'data': normal_df , 
        'color' : 'rgb(204, 204, 204)' 
    }

    data['usermark'] = { 
        'data': usermark_df, 
        'color': 'rgb(255, 99, 71)'
    }

    data['fiducial'] = { 
        'data': fiducial_df, 
        'color': 'rgb(0, 153, 25)'
    }

    data['side'] = { 
        'data': side_df, 
        'color': 'rgb(102, 102, 255)'
    }

    data['board'] = { 
        'data': board_df, 
        'color': 'rgb(26, 230, 230)'
    }

    for option in data.keys(): 
        tl_x = data[option]['data'].tl_x.to_numpy()
        tl_y = data[option]['data'].tl_y.to_numpy()
        br_x = data[option]['data'].br_x.to_numpy()
        br_y = data[option]['data'].br_y.to_numpy()

        pad = np.array([None] * len(tl_x))

        x = np.stack([tl_x, tl_x, br_x, br_x, pad], axis=1)
        y = np.stack([tl_y, br_y, br_y, tl_y, pad], axis=1)

        data[option]['x'] = x.flatten()
        data[option]['y'] = y.flatten() 

    return data

def get_fov_rect(df, size_df): 

    fov_size = (size_df.fov_size - size_df.margin_size).to_numpy() 
    side_Fov_size = (size_df.side_fov_size - size_df.side_margin_size).to_numpy() 
    fov_rect_list = [] 
    fov_center_list = {'x': [], 'y': []}

    for i, row in df.iterrows(): 

        fov_center_list['x'].append(row['x']) 
        fov_center_list['y'].append(row['y']) 

        x1 = row['x'] - fov_size[0] / 2 
        y1 = row['y'] - fov_size[1] / 2 
        x2 = row['x'] + fov_size[0] / 2 
        y2 = row['y'] + fov_size[1] / 2

        x = [x1, x1, x2, x2, x1] 
        y = [y1, y2, y2, y1, y1] 

        if row['type'] == 7: 
            color = 'rgb(0, 153, 25)' 
        elif row['type'] == 4: 
            color = 'rgb(255, 90, 71)'
        elif row['side'] == 1: 
            color = 'rgb(102, 102, 255)'
        elif row['board'] == 1: 
            color = 'rgb(26, 230, 230)'
        else: 
            color = 'rgb(204, 204, 204)'

        fov_rect = {'x': x, 'y': y, 'color': color} 
        fov_rect_list.append(fov_rect) 

    return fov_center_list, fov_rect_list

def get_size_rect(): 
    pass 

if __name__ == '__main__': 

    csv_folder = 'csv' 
    root_folder = 'simulation_data' 

    for sub_folder in os.listdir(csv_folder): 

        component_file = os.path.join(csv_folder, sub_folder, 'component.csv') 
        fov_file = os.path.join(csv_folder, sub_folder, 'output_fov_ipo_1.csv') 
        size_file = os.path.join(csv_folder, sub_folder, 'size.csv') 

        component_df = pd.read_csv(component_file) 
        component_rect = get_component_rect(component_df)

        fov_df = pd.read_csv(fov_file)
        fov_df.drop(['sub_type', 'step', 'big', 'array', 'z', 'array_idx'], axis=1, inplace=True)

        fov_df['type'] = fov_df['type'].replace({4: 1, 7: 2})
        fov_df.to_csv(os.path.join(root_folder, sub_folder, 'fov.csv')) 

