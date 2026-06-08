import numpy as np
import pandas as pd
import json
import os
import math

cimport numpy as np


def solve(str data_folder, int ntrial=8):

    component = pd.read_csv(os.path.join(data_folder, 'input_component.csv'))
    size = pd.read_csv(os.path.join(data_folder, 'input_size.csv'))

    cdef double pcb_w = float(size['pcb_size'].iloc[0])
    cdef double pcb_h = float(size['pcb_size'].iloc[1])
    cdef double fov_w = float(size['fov_size'].iloc[0])
    cdef double fov_h = float(size['fov_size'].iloc[1])
    cdef int n = len(component)

    tl_x = component['tl_x'].values.astype(np.float64)
    tl_y = component['tl_y'].values.astype(np.float64)
    br_x = component['br_x'].values.astype(np.float64)
    br_y = component['br_y'].values.astype(np.float64)
    comp_type = component['type'].values.astype(np.int32)
    comp_side = component['side'].values.astype(np.int32)
    comp_step = component['step'].values.astype(np.int32)
    cdef int max_step = int(comp_step.max()) if n > 0 else 0

    # complement coding: X[i] = [[tl_x, pcb_w-br_x], [tl_y, pcb_h-br_y], [step, max_step-step]]
    X = np.zeros((n, 3, 2), dtype=np.float64)
    cdef int i
    for i in range(n):
        X[i, 0, 0] = tl_x[i]
        X[i, 0, 1] = pcb_w - br_x[i]
        X[i, 1, 0] = tl_y[i]
        X[i, 1, 1] = pcb_h - br_y[i]
        X[i, 2, 0] = <double>comp_step[i]
        X[i, 2, 1] = <double>(max_step - comp_step[i])

    # vigilance parameter
    cdef double rho_x = pcb_w - fov_w
    cdef double rho_y = pcb_h - fov_h
    cdef double rho_z = <double>max_step - 0.5

    # sort: fiducials first, then by distance from origin
    cdef list sort_order = []
    for i in range(n):
        dist = tl_x[i] + tl_y[i]
        sort_order.append((int(comp_type[i] != 0), -int(comp_type[i]), dist, i))
    sort_order.sort(key=lambda t: (-t[0], t[1], t[2]))
    base_order = np.array([t[3] for t in sort_order], dtype=np.int32)

    # multi-trial: collect all cluster centers as MILP candidates
    cdef list all_clusters = []
    cdef list best_clusters = None
    cdef int best_count = n + 1
    cdef int trial
    cdef int ci, ix, iy, n_tile_x, n_tile_y, fi
    cdef double comp_w, comp_h, tcx, tcy, rgn_tl_x, rgn_tl_y, rgn_br_x, rgn_br_y
    cdef double margin = 0.1, left_cx, right_cx, top_cy, bot_cy
    cdef bint covered

    for trial in range(ntrial):
        order = _get_trial_order(base_order, n, trial)
        clusters = _run_one_trial(X, comp_type, comp_side, order, n, rho_x, rho_y, rho_z, pcb_w, pcb_h, fov_w, fov_h)
        _expand_all(clusters, rho_x, rho_y, rho_z)
        norm_count = sum(1 for c in clusters if c['type'] == 0)
        if norm_count < best_count:
            best_count = norm_count
            best_clusters = clusters
        all_clusters.extend(clusters)

    # identify big components
    big_comps = set()
    for ci in range(n):
        if (br_x[ci] - tl_x[ci]) > fov_w or (br_y[ci] - tl_y[ci]) > fov_h:
            big_comps.add(ci)

    # fiducial + usermark FOVs from best trial (sorted: fiducial=2 before usermark=1)
    cdef list fid_fovs = []
    cdef double cx, cy
    for c in best_clusters:
        if c['type'] == 0:
            continue
        cx = (c['fov'][0, 0] + pcb_w - c['fov'][0, 1]) / 2.0
        cy = (c['fov'][1, 0] + pcb_h - c['fov'][1, 1]) / 2.0
        cx = max(fov_w / 2.0, min(pcb_w - fov_w / 2.0, cx))
        cy = max(fov_h / 2.0, min(pcb_h - fov_h / 2.0, cy))
        fid_fovs.append((round(cx, 2), round(cy, 2), list(c['idx']), c['type']))
    fid_fovs.sort(key=lambda f: -f[3])

    # extract normal candidate centers from all trials + deduplicate
    cdef list norm_candidates = []
    for c in all_clusters:
        if c['type'] != 0:
            continue
        cx = (c['fov'][0, 0] + pcb_w - c['fov'][0, 1]) / 2.0
        cy = (c['fov'][1, 0] + pcb_h - c['fov'][1, 1]) / 2.0
        cx = max(fov_w / 2.0, min(pcb_w - fov_w / 2.0, cx))
        cy = max(fov_h / 2.0, min(pcb_h - fov_h / 2.0, cy))
        cand_step = comp_step[c['idx'][0]]
        dup = False
        for dcx, dcy, ds in norm_candidates:
            if abs(cx - dcx) < 1.0 and abs(cy - dcy) < 1.0 and cand_step == ds:
                dup = True
                break
        if not dup:
            norm_candidates.append((round(cx, 2), round(cy, 2), int(cand_step)))

    # MILP set cover for normal components
    from ky_solver_origin.milp_cover import solve_milp_cover
    milp_result = solve_milp_cover(
        norm_candidates, tl_x, tl_y, br_x, br_y,
        comp_step, comp_side, comp_type,
        fov_w, fov_h, big_comps)

    if milp_result is not None:
        ordered = fid_fovs + milp_result
    else:
        norm_fovs = []
        for c in best_clusters:
            if c['type'] != 0:
                continue
            cx = (c['fov'][0, 0] + pcb_w - c['fov'][0, 1]) / 2.0
            cy = (c['fov'][1, 0] + pcb_h - c['fov'][1, 1]) / 2.0
            cx = max(fov_w / 2.0, min(pcb_w - fov_w / 2.0, cx))
            cy = max(fov_h / 2.0, min(pcb_h - fov_h / 2.0, cy))
            norm_fovs.append((round(cx, 2), round(cy, 2), list(c['idx']), 0))
        ordered = fid_fovs + _nearest_neighbor_order(norm_fovs)

    # big components: tile with FOVs to guarantee full coverage
    for ci in range(n):
        comp_w = br_x[ci] - tl_x[ci]
        comp_h = br_y[ci] - tl_y[ci]
        if comp_w <= fov_w and comp_h <= fov_h:
            continue

        # remove from ART-assigned FOVs first
        for fi in range(len(ordered)):
            if ci in ordered[fi][2]:
                ordered[fi][2].remove(ci)

        n_tile_x = max(1, int(math.ceil(comp_w / fov_w)))
        n_tile_y = max(1, int(math.ceil(comp_h / fov_h)))

        for ix in range(n_tile_x):
            for iy in range(n_tile_y):
                if n_tile_x == 1:
                    tcx = (tl_x[ci] + br_x[ci]) / 2.0
                else:
                    left_cx = tl_x[ci] + fov_w / 2.0 - margin
                    right_cx = br_x[ci] - fov_w / 2.0 + margin
                    tcx = left_cx + ix * (right_cx - left_cx) / (n_tile_x - 1)
                if n_tile_y == 1:
                    tcy = (tl_y[ci] + br_y[ci]) / 2.0
                else:
                    top_cy = tl_y[ci] + fov_h / 2.0 - margin
                    bot_cy = br_y[ci] - fov_h / 2.0 + margin
                    tcy = top_cy + iy * (bot_cy - top_cy) / (n_tile_y - 1)
                tcx = max(fov_w / 2.0, min(pcb_w - fov_w / 2.0, tcx))
                tcy = max(fov_h / 2.0, min(pcb_h - fov_h / 2.0, tcy))

                rgn_tl_x = max(tl_x[ci], tcx - fov_w / 2.0)
                rgn_tl_y = max(tl_y[ci], tcy - fov_h / 2.0)
                rgn_br_x = min(br_x[ci], tcx + fov_w / 2.0)
                rgn_br_y = min(br_y[ci], tcy + fov_h / 2.0)

                covered = False
                for fi in range(len(ordered)):
                    fov_comp_list = ordered[fi][2]
                    if len(fov_comp_list) > 0 and comp_step[fov_comp_list[0]] != comp_step[ci]:
                        continue
                    cx_f = ordered[fi][0]
                    cy_f = ordered[fi][1]
                    if (cx_f - fov_w / 2.0 <= rgn_tl_x + 1e-6 and cx_f + fov_w / 2.0 >= rgn_br_x - 1e-6 and
                        cy_f - fov_h / 2.0 <= rgn_tl_y + 1e-6 and cy_f + fov_h / 2.0 >= rgn_br_y - 1e-6):
                        if ci not in fov_comp_list:
                            fov_comp_list.append(ci)
                        covered = True
                        break

                if not covered:
                    ordered.append((round(tcx, 2), round(tcy, 2), [ci], 0))

    # remove empty FOVs
    ordered = [f for f in ordered if len(f[2]) > 0]

    # shift normal FOV centers towards PCB center (reduce travel distance)
    cdef double pcb_cx = pcb_w / 2.0, pcb_cy = pcb_h / 2.0
    cdef double cx_lo, cx_hi, cy_lo, cy_hi, hw, hh
    for fi in range(len(ordered)):
        if ordered[fi][3] != 0:
            continue
        comp_list = ordered[fi][2]
        if not comp_list:
            continue
        if any(c in big_comps for c in comp_list):
            continue
        cx_lo = -1e18
        cx_hi = 1e18
        cy_lo = -1e18
        cy_hi = 1e18
        for c in comp_list:
            if comp_side[c] == 1:
                hw = fov_w / 4.0
                hh = fov_h / 4.0
            else:
                hw = fov_w / 2.0
                hh = fov_h / 2.0
            if br_x[c] - hw > cx_lo:
                cx_lo = br_x[c] - hw
            if tl_x[c] + hw < cx_hi:
                cx_hi = tl_x[c] + hw
            if br_y[c] - hh > cy_lo:
                cy_lo = br_y[c] - hh
            if tl_y[c] + hh < cy_hi:
                cy_hi = tl_y[c] + hh
        cx_lo = max(cx_lo, fov_w / 2.0)
        cx_hi = min(cx_hi, pcb_w - fov_w / 2.0)
        cy_lo = max(cy_lo, fov_h / 2.0)
        cy_hi = min(cy_hi, pcb_h - fov_h / 2.0)
        new_cx = max(cx_lo, min(cx_hi, pcb_cx))
        new_cy = max(cy_lo, min(cy_hi, pcb_cy))
        ordered[fi] = (round(new_cx, 2), round(new_cy, 2), comp_list, ordered[fi][3])

    # V2 beam search reorder
    final_fid = [f for f in ordered if f[3] != 0]
    final_norm = [f for f in ordered if f[3] == 0]

    if len(final_norm) >= 3:
        from ky_solver.v2_planning import v2_beam_search
        param = pd.read_csv(os.path.join(data_folder, 'input_parameter.csv'))
        p_v_x = float(param['v_x'].iloc[0])
        p_v_y = float(param['v_y'].iloc[0])
        p_a_x = float(param['a_x'].iloc[0])
        p_a_y = float(param['a_y'].iloc[0])
        p_capture_time = float(param['capture_time'].iloc[0])
        p_recon_time = float(param['recon_time'].iloc[0])
        p_side_capture_time = float(param['side_capture_time'].iloc[0])

        comp_time_arr = component['time'].values.astype(np.float64)
        fov_centers_arr = np.array([[f[0], f[1]] for f in final_norm], dtype=np.float64)

        fov_imaging_arr = np.zeros(len(final_norm), dtype=np.float64)
        for _vi in range(len(final_norm)):
            _has_side = False
            for _ci in final_norm[_vi][2]:
                if comp_side[_ci] == 1:
                    _has_side = True
                    break
            fov_imaging_arr[_vi] = max(p_capture_time, p_side_capture_time) if _has_side else p_capture_time

        fov_recon_arr = np.full(len(final_norm), p_recon_time, dtype=np.float64)

        fov_comp_arr = np.zeros(len(final_norm), dtype=np.float64)
        fov_steps_arr = np.zeros(len(final_norm), dtype=np.int32)
        for _vi in range(len(final_norm)):
            comp_list = final_norm[_vi][2]
            if len(comp_list) > 0:
                _max_ct = 0.0
                for _ci in comp_list:
                    if comp_time_arr[_ci] > _max_ct:
                        _max_ct = comp_time_arr[_ci]
                fov_comp_arr[_vi] = _max_ct
                fov_steps_arr[_vi] = comp_step[comp_list[0]]

        try:
            v2_results = v2_beam_search(
                fov_centers_arr.astype(np.float32),
                fov_imaging_arr.astype(np.float32),
                fov_recon_arr.astype(np.float32),
                fov_comp_arr.astype(np.float32),
                fov_steps_arr,
                p_v_x, p_v_y, p_a_x, p_a_y,
                'open', topk=48)
            ordered = final_fid + [final_norm[i] for i in v2_results[0]]
        except Exception:
            ordered = final_fid + final_norm
    else:
        ordered = final_fid + final_norm

    # find minimum core count where CT plateaus
    sim_param = pd.read_csv(os.path.join(data_folder, 'input_parameter.csv'))
    sim_v_x = float(sim_param['v_x'].iloc[0])
    sim_v_y = float(sim_param['v_y'].iloc[0])
    sim_a_x = float(sim_param['a_x'].iloc[0])
    sim_a_y = float(sim_param['a_y'].iloc[0])
    sim_cap = float(sim_param['capture_time'].iloc[0])
    sim_rec = float(sim_param['recon_time'].iloc[0])
    sim_side = float(sim_param['side_capture_time'].iloc[0])
    max_core_input = int(sim_param['max_core'].iloc[0])

    sim_comp_time = component['time'].values.astype(np.float64)
    sim_t_img = []
    sim_t_comp = []
    sim_fov_steps = []
    for _fi in range(len(ordered)):
        _cl = ordered[_fi][2]
        _has_s = any(comp_side[c] == 1 for c in _cl)
        sim_t_img.append(max(sim_cap, sim_side) if _has_s else sim_cap)
        sim_t_comp.append([sim_comp_time[c] for c in _cl])
        sim_fov_steps.append(int(comp_step[_cl[0]]) if _cl else 0)

    ct_prev = 1e18
    optimal_core = 4
    for _nc in range(4, max(max_core_input + 1, 5)):
        ct_sim = _simulate_ct(
            ordered, sim_t_img, sim_rec, sim_t_comp, sim_fov_steps,
            sim_v_x, sim_v_y, sim_a_x, sim_a_y, _nc)
        if ct_sim >= ct_prev:
            break
        ct_prev = ct_sim
        optimal_core = _nc

    sim_param.loc[:, 'max_core'] = optimal_core
    sim_param.to_csv(os.path.join(data_folder, 'input_parameter.csv'), index=False)

    # save
    rows = []
    for cx, cy, comp_list, _ in ordered:
        rows.append({'x': round(cx, 2), 'y': round(cy, 2), 'comp_idx': json.dumps(comp_list)})

    fov_df = pd.DataFrame(rows)
    fov_df.to_csv(os.path.join(data_folder, 'output_fov.csv'), index=True)

    return len(ordered)


cdef double _simulate_ct(list fovs, list t_img, double t_recon, list t_comp,
                          list fov_steps, double v_x, double v_y, double a_x, double a_y,
                          int max_core):
    cdef int seqlen = len(fovs)
    if seqlen == 0:
        return 0.0

    cdef double th_x = v_x * v_x / a_x
    cdef double th_y = v_y * v_y / a_y

    cdef np.ndarray[np.float64_t, ndim=1] t_last = np.zeros(max_core, dtype=np.float64)
    cdef double dx, dy, dy_t, t_move, t_img_start, sel_t
    cdef int i, j, k, rc, cid

    # first FOV: capture
    t_last[0] = t_img[0]

    # first FOV: recon on core 1 or 2
    rc = 1 if t_last[1] <= t_last[2] else 2
    t_last[rc] = max(t_last[rc], t_last[0]) + t_recon
    sel_t = t_last[rc]

    # first FOV: component inspection on core 3+
    for j in range(len(t_comp[0])):
        cid = 3
        for k in range(4, max_core):
            if max(t_last[k], sel_t) < max(t_last[cid], sel_t):
                cid = k
        t_last[cid] = max(t_last[cid], sel_t) + t_comp[0][j]

    for i in range(seqlen - 1):
        # motion time (max of x-axis and y-axis)
        dx = abs(fovs[i + 1][0] - fovs[i][0])
        dy = abs(fovs[i + 1][1] - fovs[i][1])
        if dx >= th_x:
            t_move = 2.0 * v_x / a_x + (dx - th_x) / v_x
        else:
            t_move = 2.0 * math.sqrt(dx / a_x) if dx > 0 else 0.0
        dy_t = (2.0 * v_y / a_y + (dy - th_y) / v_y) if dy >= th_y else (2.0 * math.sqrt(dy / a_y) if dy > 0 else 0.0)
        if dy_t > t_move:
            t_move = dy_t

        t_last[0] += t_move

        # step transition delay
        if fov_steps[i] != fov_steps[i + 1]:
            t_last[0] += 5.0

        # imaging
        t_img_start = t_last[0]
        t_last[0] = t_img_start + t_img[i + 1]

        # recon on core 1 or 2
        for k in range(max_core):
            if t_last[k] < t_last[0]:
                t_last[k] = t_last[0]
        rc = 1 if t_last[1] <= t_last[2] else 2
        t_last[rc] += t_recon
        sel_t = t_last[rc]

        # component inspection on core 3+
        for j in range(len(t_comp[i + 1])):
            cid = 3
            for k in range(4, max_core):
                if max(t_last[k], sel_t) < max(t_last[cid], sel_t):
                    cid = k
            t_last[cid] = max(t_last[cid], sel_t) + t_comp[i + 1][j]

    return float(np.max(t_last))


cdef list _get_trial_order(np.ndarray[np.int32_t, ndim=1] base_order, int n, int trial):
    cdef list order = list(base_order)
    if trial == 0:
        return order

    cdef int stride = 16 + trial * 3
    cdef list reordered = list(order)
    cdef int i_, i
    for i_ in range(n):
        if i_ < n - n % (stride + 1):
            i = i_ + stride - 2 * (i_ % (stride + 1))
            if 0 <= i < n:
                reordered[i_] = order[i]
            else:
                reordered[i_] = order[i_]
        else:
            reordered[i_] = order[i_]
    return reordered


cdef list _run_one_trial(np.ndarray[np.float64_t, ndim=3] X,
                          np.ndarray[np.int32_t, ndim=1] comp_type,
                          np.ndarray[np.int32_t, ndim=1] comp_side,
                          list order, int n,
                          double rho_x, double rho_y, double rho_z,
                          double pcb_w, double pcb_h,
                          double fov_w, double fov_h):

    cdef list clusters = []
    cdef np.ndarray[np.int32_t, ndim=1] visited = np.full(n, -1, dtype=np.int32)
    cdef int i, idx, cluster_idx
    cdef double eff_rho_x = rho_x
    cdef double eff_rho_y = rho_y

    # side vigilance (tighter: side_fov = fov/2)
    cdef double side_fov_w = fov_w / 2.0
    cdef double side_fov_h = fov_h / 2.0
    cdef double side_rho_x = pcb_w - side_fov_w
    cdef double side_rho_y = pcb_h - side_fov_h

    # side components first — use side_rho (tighter constraint)
    for idx in order:
        i = int(idx)
        if comp_side[i] != 1 or visited[i] >= 0:
            continue
        cluster_idx = _check_update(clusters, X[i], side_rho_x, side_rho_y, rho_z)
        if cluster_idx == -1:
            _enroll(clusters, X[i], i, comp_type[i], comp_side[i])
            visited[i] = len(clusters) - 1
        else:
            _update(clusters[cluster_idx], X[i], side_rho_x, side_rho_y, rho_z, i)
            visited[i] = cluster_idx

    # expand side clusters for top-cam inclusion
    for c in clusters:
        for d in range(2):
            rho_d = eff_rho_x if d == 0 else eff_rho_y
            side_rho_d = side_rho_x if d == 0 else side_rho_y
            complement = c['fov'][d, 0] + c['fov'][d, 1]
            slack = max(rho_d, rho_d - side_rho_d + complement)
            delta = c['fov'][d, 0] + c['fov'][d, 1] - slack
            if delta > 0:
                c['fov'][d, 0] -= delta / 2.0 - 1e-5
                c['fov'][d, 1] -= delta / 2.0 - 1e-5

    # fiducial + remaining components
    for idx in order:
        i = int(idx)
        if visited[i] >= 0:
            continue

        x = X[i].copy()
        if comp_type[i] == 1:
            x = _expand_single(x, eff_rho_x, eff_rho_y)

        cluster_idx = _check_update(clusters, x, eff_rho_x, eff_rho_y, rho_z)
        if cluster_idx == -1:
            fov = x.copy()
            if comp_type[i] == 1:
                fov = _expand_single(fov, eff_rho_x, eff_rho_y)
            _enroll(clusters, fov, i, comp_type[i], comp_side[i])
            visited[i] = len(clusters) - 1
        else:
            _update(clusters[cluster_idx], x, eff_rho_x, eff_rho_y, rho_z, i)
            visited[i] = cluster_idx
            if comp_type[i] > clusters[cluster_idx]['type']:
                clusters[cluster_idx]['type'] = comp_type[i]

    return clusters


cdef int _check_update(list clusters, np.ndarray[np.float64_t, ndim=2] x,
                        double rho_x, double rho_y, double rho_z):
    if len(clusters) == 0:
        return -1

    cdef int idx = -1
    cdef double min_diff = 1e18
    cdef double h_next, w_next, diff
    cdef int i

    for i in range(len(clusters)):
        c = clusters[i]
        fov = c['fov']

        # element-wise minimum
        fov_min_0_0 = min(fov[0, 0], x[0, 0])
        fov_min_0_1 = min(fov[0, 1], x[0, 1])
        fov_min_1_0 = min(fov[1, 0], x[1, 0])
        fov_min_1_1 = min(fov[1, 1], x[1, 1])
        fov_min_2_0 = min(fov[2, 0], x[2, 0])
        fov_min_2_1 = min(fov[2, 1], x[2, 1])

        # vigilance test
        if fov_min_0_0 + fov_min_0_1 < rho_x:
            continue
        if fov_min_1_0 + fov_min_1_1 < rho_y:
            continue
        if fov_min_2_0 + fov_min_2_1 < rho_z:
            continue

        # expansion cost (x + y only, step is binary match)
        h_next = (fov[0, 0] + fov[0, 1]) - (fov_min_0_0 + fov_min_0_1)
        w_next = (fov[1, 0] + fov[1, 1]) - (fov_min_1_0 + fov_min_1_1)
        diff = h_next + w_next

        if diff < min_diff:
            min_diff = diff
            idx = i
        elif diff == min_diff and idx >= 0:
            if len(c['idx']) < len(clusters[idx]['idx']):
                idx = i

    return idx


cdef void _update(dict cluster, np.ndarray[np.float64_t, ndim=2] x,
                   double rho_x, double rho_y, double rho_z, int comp_idx):

    cdef np.ndarray[np.float64_t, ndim=2] fov = cluster['fov']
    cdef double r1, r2, beta, rho_d
    cdef int d

    for d in range(3):
        if d == 0:
            rho_d = rho_x
        elif d == 1:
            rho_d = rho_y
        else:
            rho_d = rho_z
        r1 = fov[d, 0] + fov[d, 1]
        min_0 = min(fov[d, 0], x[d, 0])
        min_1 = min(fov[d, 1], x[d, 1])
        r2 = min_0 + min_1

        beta = 0.0
        if r2 < rho_d <= r1:
            beta = (rho_d - r2) / (r1 - r2)

        fov[d, 0] = beta * fov[d, 0] + (1.0 - beta) * min_0
        fov[d, 1] = beta * fov[d, 1] + (1.0 - beta) * min_1

    cluster['idx'].append(comp_idx)


cdef void _enroll(list clusters, np.ndarray[np.float64_t, ndim=2] fov,
                   int comp_idx, int comp_type, int comp_side):
    cdef dict c = {
        'fov': fov.copy(),
        'type': comp_type,
        'side': comp_side,
        'idx': [comp_idx],
    }
    clusters.append(c)


cdef np.ndarray[np.float64_t, ndim=2] _expand_single(np.ndarray[np.float64_t, ndim=2] x,
                                                        double rho_x, double rho_y):
    cdef np.ndarray[np.float64_t, ndim=2] out = x.copy()
    cdef double delta, rho_d
    cdef int d
    for d in range(2):
        rho_d = rho_x if d == 0 else rho_y
        delta = out[d, 0] + out[d, 1] - rho_d
        if delta > 0:
            out[d, 0] -= delta / 2.0 - 1e-5
            out[d, 1] -= delta / 2.0 - 1e-5
    return out


cdef void _expand_all(list clusters, double rho_x, double rho_y, double rho_z):
    cdef double margin = -0.001
    cdef double delta, rho_d
    cdef int d

    for c in clusters:
        fov = c['fov']
        for d in range(2):
            rho_d = rho_x if d == 0 else rho_y
            delta = fov[d, 0] + fov[d, 1] - rho_d
            if delta > 0:
                fov[d, 0] -= delta / 2.0
                fov[d, 1] -= delta / 2.0

                if fov[d, 0] < margin:
                    fov[d, 1] += (fov[d, 0] - margin)
                    fov[d, 0] = margin
                if fov[d, 1] < margin:
                    fov[d, 0] += (fov[d, 1] - margin)
                    fov[d, 1] = margin


cdef list _nearest_neighbor_order(list fovs):
    if len(fovs) <= 1:
        return fovs

    cdef list remaining = list(range(len(fovs)))
    cdef list ordered = []
    cdef int current = 0
    cdef double cx, cy, dx, dy, dist, best_dist
    cdef int best_idx, idx

    remaining.remove(current)
    ordered.append(fovs[current])

    while remaining:
        cx = ordered[-1][0]
        cy = ordered[-1][1]
        best_dist = 1e18
        best_idx = remaining[0]

        for idx in remaining:
            dx = fovs[idx][0] - cx
            dy = fovs[idx][1] - cy
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        remaining.remove(best_idx)
        ordered.append(fovs[best_idx])

    return ordered
