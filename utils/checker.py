import os
import json
import math 
import logging
import seaborn as sns
import numpy as np
import pandas as pd 
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class Checker:
    def __init__(self, folder): 

        self.folder = folder 

        # log information
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{folder}")
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(f"[{self.folder}] %(message)s")

        if os.path.exists(os.path.join(self.folder, 'error.log')): 
            os.remove(os.path.join(self.folder, 'error.log'))
        file_handler = logging.FileHandler(os.path.join(self.folder, 'error.log'))
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

        # check files and load 
        component_file = os.path.join(folder, 'component.csv')
        if not os.path.isfile(component_file): 
            assert False, f'No component.csv file at {component_file}'

        size_file = os.path.join(folder, 'size.csv')
        if not os.path.isfile(size_file): 
            assert False, f'No size.csv file at {size_file}' 

        parameter_file = os.path.join(self.folder, 'parameter.csv') 
        if not os.path.isfile(parameter_file): 
            assert False, f'No parameter.csv file at {parameter_file}' 

        fov_file = os.path.join(folder, 'fov.csv') 
        if not os.path.isfile(fov_file): 
            assert False, f'No fov.csv file at {fov_file}' 

        self.component = pd.read_csv(component_file) 
        self.parameter = pd.read_csv(parameter_file) 
        size = pd.read_csv(size_file) 

        self.pcb_width = size['pcb_size'].iloc[0] 
        self.pcb_height = size['pcb_size'].iloc[1] 

        fov_size = size['fov_size'] 
        self.fov_width = fov_size.iloc[0]
        self.fov_height = fov_size.iloc[1]

        self.fov = pd.read_csv(fov_file) 

        # get fov rect 
        self.fov['tl_x'] = self.fov['x'] - self.fov_width / 2
        self.fov['tl_y'] = self.fov['y'] - self.fov_height / 2
        self.fov['br_x'] = self.fov['x'] + self.fov_width / 2
        self.fov['br_y'] = self.fov['y'] + self.fov_height / 2

        # update fov type 
        fov_type = [] 
        for i, fov in self.fov.iterrows(): 
            comp_idx = json.loads(self.fov.iloc[i]['comp_idx'])
            max_type = 0 
            for idx in comp_idx: 
                if idx < len(self.component):
                    max_type = max(max_type, self.component.iloc[idx]['type']) 

            fov_type.append(max_type) 

        self.fov['type'] = fov_type 

    def is_target_fully_covered(self, rects, target):
        tx1, ty1, tx2, ty2 = target

        # Step 1: Filter rectangles that intersect the target area
        filtered = []
        x_edges = {tx1, tx2}
        y_edges = {ty1, ty2}

        for x1, y1, x2, y2 in rects:
            # Skip rectangles completely outside the target
            if x2 <= tx1 or x1 >= tx2 or y2 <= ty1 or y1 >= ty2:
                continue
            
            # Clip to target region
            cx1, cy1 = max(x1, tx1), max(y1, ty1)
            cx2, cy2 = min(x2, tx2), min(y2, ty2)
            filtered.append((cx1, cy1, cx2, cy2))

            x_edges.add(cx1)
            x_edges.add(cx2)
            y_edges.add(cy1)
            y_edges.add(cy2)

        # Step 2: Sort unique x and y edges to form a grid
        x_list = sorted(x_edges)
        y_list = sorted(y_edges)

        # Step 3: Check each cell in the grid is covered by at least one rectangle
        for i in range(len(x_list) - 1):
            for j in range(len(y_list) - 1):
                cx1, cx2 = x_list[i], x_list[i + 1]
                cy1, cy2 = y_list[j], y_list[j + 1]

                # Only check if this cell is inside the target
                if not (tx1 <= cx1 and cx2 <= tx2 and ty1 <= cy1 and cy2 <= ty2):
                    continue
                
                # Check if this cell is covered by at least one rectangle
                covered = any(
                    rx1 <= cx1 and rx2 >= cx2 and ry1 <= cy1 and ry2 >= cy2
                    for rx1, ry1, rx2, ry2 in filtered
                )
                if not covered:
                    return False  # Uncovered cell found

        return True  # All subcells covered

    def cover_all_components(self): 

        # find big comp
        comp_big_mask = [0] * len(self.component) 
        comp_mask = [0] * len(self.component) 

        error_component_list = [] 

        big_comp_idx = dict() 
        for i, comp in self.component.iterrows(): 
                tl_x = comp['tl_x'] 
                tl_y = comp['tl_y'] 
                br_x = comp['br_x'] 
                br_y = comp['br_y'] 

                n_width, n_height = 1, 1

                if (br_x - tl_x) > self.fov_width:
                    n_width = math.ceil((br_x - tl_x) / self.fov_width)

                if (br_y - tl_y) > self.fov_height: 
                    n_height = math.ceil((br_y - tl_y) / self.fov_height)

                if n_width > 1 or n_height > 1: 
                    comp_big_mask[i] = 1
                    big_comp_idx[i] = [] 

        for i, fov in self.fov.iterrows(): 
            comp_idx = json.loads(fov['comp_idx'])
            for idx in comp_idx: 
                if idx in list(big_comp_idx.keys()): 
                    fov_rect = tuple(self.fov[['tl_x', 'tl_y', 'br_x', 'br_y']].iloc[i])
                    big_comp_idx[idx].append(fov_rect) 

        for k, v in big_comp_idx.items(): 
            # k: comp idx, v: list of fov rect
            comp_rect = tuple(self.component[['tl_x', 'tl_y', 'br_x', 'br_y']].iloc[k])
            success = self.is_target_fully_covered(v, comp_rect)
            if success: 
                comp_mask[k] = 1
            else:
                self.logger.error(f"{k} is out of multiple FOVs.") 
                error_component_list.append(str(k)) 
        
        success = True

        for i, fov in self.fov.iterrows(): 

            fov_tl_x = fov['tl_x'] 
            fov_tl_y = fov['tl_y']
            fov_br_x = fov['br_x'] 
            fov_br_y = fov['br_y'] 

            comp_idx = json.loads(fov['comp_idx'])
            
            for idx in comp_idx: 

                if idx >= len(self.component): 
                    self.logger.error(f'Component idx {idx} in FOV {i} is not existed.')
                    continue 

                if comp_big_mask[idx] == 1: # already considered
                    continue 

                comp_tl_x = self.component.iloc[idx]['tl_x'] 
                comp_tl_y = self.component.iloc[idx]['tl_y'] 
                comp_br_x = self.component.iloc[idx]['br_x'] 
                comp_br_y = self.component.iloc[idx]['br_y'] 

                if fov_tl_x <= comp_tl_x and fov_tl_y <= comp_tl_y and fov_br_x >= comp_br_x and fov_br_y >= comp_br_y:
                    comp_mask[idx] = 1
                else: 
                    success = False 
                    self.logger.error(f"{idx} is out of FOV.") 
                    error_component_list.append(str(idx))

                
        if sum(comp_mask) != len(self.component): 
            n_violation = len(self.component) - sum(comp_mask) 
            if n_violation > 0: 
                self.logger.error(f"{n_violation} components (" + ','.join(error_component_list) + ") is violated the conditions.") 
                success = False        

        return success 

    def fov_inspection_order(self): 
        success = all(self.fov['type'].iloc[i] >= self.fov['type'].iloc[i+1] for i in range(len(self.fov)-1))
        if not success: 
            self.logger.error('FOV Inspection order violation.') 
        return success


    def save_pcb_view(self): 

        fig = go.Figure() 

        component_rect = self.get_component_rect(self.component) 

        for option in component_rect.keys(): 
            fig.add_trace(
                go.Scatter(
                    x=component_rect[option]['x'],
                    y=component_rect[option]['y'],
                    fill='toself',
                    line=dict(width=0),
                    fillcolor=component_rect[option]['color'],
                    name=option, 
                    marker=dict(opacity=0),
                    showlegend=True
                )
            )

        # 2. Size info
        fig.add_trace(
            go.Scatter(name='PCB', 
                        x = [0, 0, self.pcb_width, self.pcb_width, 0], 
                        y = [0, self.pcb_height, self.pcb_height, 0, 0], 
                       mode='lines',
                       line=dict(color='#303030', width=1),
                       showlegend=False
                       )
        )

        # 3. Draw output
        fov_rect_list = self.get_fov_rect() 

        for i in range(len(fov_rect_list)): 

            comp_idx = json.loads(self.fov.iloc[i]['comp_idx'])
            comp_idx = [idx for idx in comp_idx if idx < len(self.component)] 
            component_rect_fov = self.get_component_rect(self.component.iloc[comp_idx])
            for option in component_rect_fov.keys(): 
                fig.add_trace(
                    go.Scatter(
                        x=component_rect_fov[option]['x'],
                        y=component_rect_fov[option]['y'],
                        fill='toself',
                        line=dict(width=0),
                        fillcolor=component_rect_fov[option]['fov_color'],
                        name=option, 
                        marker=dict(opacity=0),
                        legendgroup=f'FOV_{i}', 
                        showlegend=False
                    )
                )
            fig.add_trace(
                go.Scatter(
                    name = f'FOV_{i}', 
                    x=fov_rect_list[i]['x'], 
                    y=fov_rect_list[i]['y'], 
                    line = dict(color='#000000', width=1), 
                    fill='toself', 
                    fillcolor=fov_rect_list[i]['color'], 
                    opacity=0.3,
                    marker = dict(opacity=0), 
                    legendgroup = f'FOV_{i}', 
                    showlegend=True
                )
            )

        # FOV trajectory
        size = np.array([10] * len(self.fov)) 
        size[0] = 30 
        showlegend = True 

        for i in range(len(self.fov)-1): 
            fig.add_trace(
                go.Scatter(
                    name='FOV', x=list(self.fov.iloc[i:i+2]['x']), y=list(self.fov.iloc[i:i+2]['y']), 
                    mode='lines+markers', 
                    line=dict(color='rgb(37, 37, 37)'), 
                    marker=dict(size=[size[i], 10]), 
                    legendgroup=f'FOV', 
                    showlegend=showlegend
                )
            )

            showlegend = False 

        fig.update_layout(
            plot_bgcolor='#FFFFFF',
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )

        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, range=[0, self.pcb_width])
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, range=[self.pcb_height, 0], scaleanchor='x')

        fig.write_html(os.path.join(self.folder, 'result.html'))

    def get_fov_rect(self): 

        fov_rect_list = [] 

        for i, row in self.fov.iterrows(): 

            x1 = self.fov.iloc[i]['tl_x']
            y1 = self.fov.iloc[i]['tl_y'] 
            x2 = self.fov.iloc[i]['br_x']
            y2 = self.fov.iloc[i]['br_y'] 


            x = [x1, x1, x2, x2, x1] 
            y = [y1, y2, y2, y1, y1] 

            if row['type'] == 1: 
                color = 'rgb(0, 153, 25)' 
            else: 
                color = 'rgb(204, 204, 204)'

            fov_rect = {'x': x, 'y': y, 'color': color} 
            fov_rect_list.append(fov_rect) 

        return fov_rect_list


    def get_component_rect(self, df): 

        normal_df = df[df['type'] == 0]
        fiducial_df = df[df['type'] == 1] 

        data = dict() 
        data['normal'] = {
            'data': normal_df , 
            'color' : 'rgb(204, 204, 204)', 
            'fov_color': 'rgb(130, 130, 130)'
        }

        data['fiducial'] = { 
            'data': fiducial_df, 
            'color': 'rgb(0, 153, 25)', 
            'fov_color': 'rgb(0, 110, 10)'
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

    def save_timelog(self): 
        self.fov['imaging_time'] = self.parameter['capture_time'].iloc[0] 
        self.fov['recon_time'] = self.parameter['recon_time'].iloc[0] 
        fov_center = self.fov[['x', 'y']].to_numpy() 

        t_imaging = self.fov['imaging_time'].to_numpy()
        t_recon = self.fov['recon_time'].to_numpy()

        t_comps = [] 
        for i in range(len(self.fov)): 
            comp_idx = json.loads(self.fov.iloc[i]['comp_idx'])
            comp_idx = [idx for idx in comp_idx if idx < len(self.component)] 
            fov_component_df = self.component.iloc[comp_idx]
            t_comp = fov_component_df['time'].tolist() 
            t_comps.append(t_comp) 

        params = [self.parameter['v_x'].iloc[0], self.parameter['v_y'].iloc[0], self.parameter['a_x'].iloc[0], self.parameter['a_y'].iloc[0]]
        max_core = self.parameter['max_core'].iloc[0] 

        _, sim_ct, sim_gantt = self.get_real_cost(params,
                                                  fov_center, 
                                                  t_imaging, 
                                                  t_recon, 
                                                  t_comps, 
                                                  max_core=max_core)

        sns.set()
        fig = plt.figure(figsize=(12, 5)) 
        self.display_timelog(sim_gantt, sim_ct, 'Cycle Time')
        timelog_file = os.path.join(self.folder, f'timelog.png') 
        plt.savefig(timelog_file, bbox_inches='tight') 
        plt.close() 

        return True 

    def display_timelog(self, result, max_ct, title):

        core_pos = [0.]
        core_name = ['0']

        # probe movement
        start = list(result[(result.v_core == 0) & (result.v_key==0)].t_start)
        duration = list(result[(result.v_core == 0) & (result.v_key==0)].t_duration)
        plt.broken_barh(list(zip(start, duration)), (0, 0.8), facecolors='#1773B2', linewidth=0.2)

        # imaging
        start = list(result[(result.v_core == 0) & (result.v_key == 1)].t_start)
        duration = list(result[(result.v_core == 0) & (result.v_key == 1)].t_duration)
        plt.broken_barh(list(zip(start, duration)), (0, 0.8), facecolors='#2B9F2A', linewidth=0.2)
        plt.hlines(0, xmin=0, xmax=max_ct + 0.5, color='tab:gray', linewidth=0.1)

        for j in range(1, 3):
            start = list(result[result.v_core == j].t_start)
            duration = list(result[result.v_core == j].t_duration)

            plt.broken_barh(list(zip(start, duration)), (j, 0.8), facecolors='#E26768', linewidth=0.2)
            plt.hlines(j, xmin=0, xmax=max_ct + 0.5, color='tab:gray', linewidth=0.1)

            core_pos.append(j)
            core_name.append('%d' % j)

        max_core = max(result.v_core)

        for j in range(3, max_core + 1):
            start = list(result[result.v_core == j].t_start)
            duration = list(result[result.v_core == j].t_duration)

            plt.broken_barh(list(zip(start, duration)), (j, 0.8), facecolors='#FE9234', linewidth=0.2)
            plt.hlines(j, xmin=0, xmax=max_ct + 0.5, color='tab:gray', linewidth=0.1)
            core_pos.append(j)
            core_name.append('%d' % j)

        plt.title('Timelog (%.2fs)' % max_ct, fontdict={'fontsize': 12}) 
        plt.xticks(np.arange(0, max_ct + 1.0, max(int(max_ct/10.0), 1.0)), fontsize=10)
        plt.yticks(core_pos, []) #core_name, fontsize=0)
        plt.xlim(-0.1, max_ct + 0.5)

    def get_real_cost(self, param, point, t_img, t_fov, t_comp, max_core): 

        # total distance
        dist = np.sum(np.sqrt(np.sum((point[1:] - point[:-1])**2, axis=1)))

        vel_x, vel_y, acc_x, acc_y = param

        seqlen = len(point)

        t_check = np.zeros((seqlen)) 

        # get costmap
        point_x = point[:, 0]  # seqlen
        point_y = point[:, 1]  # seqlen

        point_x_a = np.broadcast_to(point_x[:, None], (seqlen, seqlen)) 
        point_x_b = np.broadcast_to(point_x[None, :], (seqlen, seqlen)) 

        distance_x = np.abs(point_x_a - point_x_b) 
        th_x = vel_x ** 2 / acc_x

        mask_x = (distance_x >= th_x).astype(float)

        distance_over_x = 2 * vel_x / acc_x + (distance_x - vel_x ** 2 / acc_x) / vel_x 
        distance_under_x = 2 * np.sqrt(2 * distance_x / acc_x) 

        time_x = distance_over_x * mask_x + distance_under_x * (1 - mask_x)

        point_y_a = np.broadcast_to(point_y[:, None], (seqlen, seqlen)) 
        point_y_b = np.broadcast_to(point_y[None, :], (seqlen, seqlen)) 

        distance_y = np.abs(point_y_a - point_y_b)
        th_y = vel_y ** 2 / acc_y
        mask_y = (distance_y >= th_y).astype(float) 

        distance_over_y = 2 * vel_y / acc_y + (distance_y - vel_y ** 2 / acc_y) / vel_y
        distance_under_y = 2 * np.sqrt(2 * distance_y / acc_y)

        time_y = distance_over_y * mask_y + distance_under_y * (1 - mask_y)
        cost_map = np.maximum(time_x, time_y) 

        # first fov's imaging time
        v_core = [0]
        t_start = [0]
        t_duration = [t_img[0].item()]
        v_key = [1]
        v_fov = [0] 
        t_last = np.repeat(t_img[0:1], max_core) 

        # first fov's inspection time
        temp_t_last = np.maximum(t_last, t_last[0])[1:3] 
        current_id = np.argmin(temp_t_last) + 1

        v_core.append(current_id)
        t_start.append(t_img[0]) 
        t_duration.append(t_fov[0]) 
        v_key.append(2)
        v_fov.append(0) 
        t_last[current_id] += t_fov[0]

        # first fov's components inspection time
        select_t_last = t_last[current_id]

        each_t = t_comp[0]

        for j in range(len(each_t)):
            temp_t_last = np.maximum(t_last, select_t_last) 
            temp_t_last  = temp_t_last[3:] 
            current_id = np.argmin(temp_t_last) + 3

            v_core.append(current_id)
            start_comp = max(select_t_last, t_last[current_id]) 
            t_start.append(start_comp)
            t_duration.append(each_t[j])
            v_key.append(3)
            v_fov.append(0) 
            t_last[current_id] = max(t_last[current_id], select_t_last) + each_t[j]
            t_check[0] = max(t_check[0], t_last[current_id])


        for i in range(seqlen - 1):
            # movement
            t_move = cost_map[i, i + 1]
            v_core.append(0)
            t_start.append(t_last[0]) 
            t_duration.append(t_move) 
            v_key.append(0)
            v_fov.append(i+1) 
            t_last[0] += t_move

            # imaging time
            t_check[:] = np.maximum(t_check, t_last[0]) 
            t_img_start = t_last[0]

            v_core.append(0)
            t_start.append(t_img_start)
            t_duration.append(t_img[i + 1])
            v_key.append(1)
            v_fov.append(i+1) 
            t_last[0] = t_img_start + t_img[i + 1]

            # fov inspection
            t_last[:] = np.maximum(t_last, t_last[0]) 
            temp_t_last = np.maximum(t_last, t_last[0])[1:3] 
            current_id = np.argmin(temp_t_last) + 1

            v_core.append(current_id)
            t_start.append(t_last[current_id])
            t_duration.append(t_fov[i + 1])
            v_key.append(2)
            v_fov.append(i+1) 
            t_last[current_id] += t_fov[i + 1]

            # fov's components inspection time
            select_t_last = t_last[current_id]
            each_t = t_comp[i + 1]

            for j in range(len(each_t)):

                temp_t_last = np.maximum(t_last, select_t_last) 
                temp_t_last  = temp_t_last[3:] 
                current_id = np.argmin(temp_t_last) + 3

                v_core.append(current_id)

                start_comp = max(select_t_last, t_last[current_id])
                t_start.append(start_comp)
                t_duration.append(each_t[j])
                v_key.append(3)
                v_fov.append(i+1)
                t_last[current_id] = max(t_last[current_id], select_t_last) + each_t[j]
                t_check[i+1] = max(t_check[i+1], t_last[current_id])

        result = {}
        result['t_start'] = t_start
        result['t_duration'] = t_duration
        result['v_core'] = v_core
        result['v_key'] = v_key
        result['v_fov'] = v_fov 

        result = pd.DataFrame(data=result)

        return dist, max(t_last), result
