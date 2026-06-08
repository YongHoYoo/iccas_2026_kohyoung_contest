"""
Greedy baseline solver for AOI FOV optimization.
Places FOVs one at a time using nearest-neighbor ordering,
packing as many same-step components as possible into each FOV.
"""
import os
import json
import math
import numpy as np
import pandas as pd


def solve(data_folder):
    comp = pd.read_csv(os.path.join(data_folder, 'input_component.csv'), index_col=0)
    size = pd.read_csv(os.path.join(data_folder, 'input_size.csv'))

    fov_w = float(size['fov_width'].iloc[0])
    fov_h = float(size['fov_height'].iloc[0])
    pcb_w = float(size['pcb_width'].iloc[0])
    pcb_h = float(size['pcb_height'].iloc[0])
    side_fw, side_fh = fov_w / 2.0, fov_h / 2.0

    n = len(comp)
    comp_cx = ((comp['tl_x'] + comp['br_x']) / 2.0).values
    comp_cy = ((comp['tl_y'] + comp['br_y']) / 2.0).values
    comp_w = (comp['br_x'] - comp['tl_x']).values
    comp_h = (comp['br_y'] - comp['tl_y']).values
    c_type = comp['type'].astype(int).values
    c_side = comp['side'].astype(int).values if 'side' in comp.columns else np.zeros(n, dtype=int)
    c_step = comp['step'].astype(int).values if 'step' in comp.columns else np.zeros(n, dtype=int)
    c_tl_x, c_tl_y = comp['tl_x'].values, comp['tl_y'].values
    c_br_x, c_br_y = comp['br_x'].values, comp['br_y'].values
    is_big = (comp_w > fov_w) | (comp_h > fov_h)

    assigned = [False] * n
    fovs = []

    dist = np.sqrt(comp_cx ** 2 + comp_cy ** 2)
    order = np.argsort(dist)

    # Phase 1: normal-sized components — greedy packing
    for seed_i in order:
        if assigned[seed_i] or is_big[seed_i]:
            continue

        fx = np.clip(c_tl_x[seed_i] + fov_w / 2, fov_w / 2, pcb_w - fov_w / 2)
        fy = np.clip(c_tl_y[seed_i] + fov_h / 2, fov_h / 2, pcb_h - fov_h / 2)

        f_tl_x, f_tl_y = fx - fov_w / 2, fy - fov_h / 2
        f_br_x, f_br_y = fx + fov_w / 2, fy + fov_h / 2
        s_tl_x, s_tl_y = fx - side_fw / 2, fy - side_fh / 2
        s_br_x, s_br_y = fx + side_fw / 2, fy + side_fh / 2

        seed_step = int(c_step[seed_i])
        covered = []

        for j in order:
            if assigned[j] or is_big[j]:
                continue
            if c_step[j] != seed_step:
                continue
            if c_side[j] == 1:
                bl_x, bl_y, br_x2, br_y2 = s_tl_x, s_tl_y, s_br_x, s_br_y
            else:
                bl_x, bl_y, br_x2, br_y2 = f_tl_x, f_tl_y, f_br_x, f_br_y
            tol = 1e-5
            if (bl_x - tol <= c_tl_x[j] and bl_y - tol <= c_tl_y[j] and
                    br_x2 + tol >= c_br_x[j] and br_y2 + tol >= c_br_y[j]):
                covered.append(j)

        if covered:
            for j in covered:
                assigned[j] = True
            fovs.append({'x': fx, 'y': fy, 'comp_idx': covered, 'step': seed_step})

    # Phase 2: big components — grid FOVs
    for i in range(n):
        if assigned[i]:
            continue
        cw, ch = comp_w[i], comp_h[i]
        n_x = math.ceil(cw / fov_w) if cw > fov_w else 1
        n_y = math.ceil(ch / fov_h) if ch > fov_h else 1
        if n_x == 1 and n_y == 1:
            fx = np.clip(comp_cx[i], fov_w / 2, pcb_w - fov_w / 2)
            fy = np.clip(comp_cy[i], fov_h / 2, pcb_h - fov_h / 2)
            fovs.append({'x': fx, 'y': fy, 'comp_idx': [i], 'step': int(c_step[i])})
        else:
            for gx in range(n_x):
                for gy in range(n_y):
                    fx = c_tl_x[i] + fov_w / 2 + gx * fov_w
                    fy = c_tl_y[i] + fov_h / 2 + gy * fov_h
                    fx = min(fx, c_br_x[i] - fov_w / 2)
                    fy = min(fy, c_br_y[i] - fov_h / 2)
                    fovs.append({'x': fx, 'y': fy, 'comp_idx': [i], 'step': int(c_step[i])})
        assigned[i] = True

    # Compute FOV type from components
    for f in fovs:
        f['type'] = max((int(c_type[j]) for j in f['comp_idx']), default=0)

    # Sort by type descending (Fiducial=2 → Barcode=1 → Normal=0)
    fovs.sort(key=lambda f: -f['type'])

    # Nearest-neighbor ordering within same type groups
    ordered = []
    type_groups = {}
    for f in fovs:
        type_groups.setdefault(f['type'], []).append(f)

    lx, ly = 0.0, 0.0
    for t in sorted(type_groups.keys(), reverse=True):
        remaining = list(type_groups[t])
        while remaining:
            best = min(range(len(remaining)),
                       key=lambda j: (remaining[j]['x'] - lx) ** 2 + (remaining[j]['y'] - ly) ** 2)
            chosen = remaining.pop(best)
            ordered.append(chosen)
            lx, ly = chosen['x'], chosen['y']

    # Write output
    rows = []
    for f in ordered:
        rows.append({
            'x': f['x'], 'y': f['y'],
            'comp_idx': json.dumps([int(x) for x in f['comp_idx']])
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(data_folder, 'output_fov.csv'))

    return len(df)
