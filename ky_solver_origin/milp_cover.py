import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix


def solve_milp_cover(candidates, tl_x, tl_y, br_x, br_y,
                     comp_step, comp_side, comp_type,
                     fov_w, fov_h, big_comps):
    n_comps = len(tl_x)
    normal_comps = [i for i in range(n_comps)
                    if i not in big_comps and comp_type[i] == 0]
    if not normal_comps or not candidates:
        return None

    n_cand = len(candidates)
    n_norm = len(normal_comps)

    rows, cols = [], []
    for j, (cx, cy, step) in enumerate(candidates):
        ftl_x = cx - fov_w * 0.5
        ftl_y = cy - fov_h * 0.5
        fbr_x = cx + fov_w * 0.5
        fbr_y = cy + fov_h * 0.5
        stl_x = cx - fov_w * 0.25
        stl_y = cy - fov_h * 0.25
        sbr_x = cx + fov_w * 0.25
        sbr_y = cy + fov_h * 0.25

        for ii, ci in enumerate(normal_comps):
            if comp_step[ci] != step:
                continue
            if comp_side[ci] == 1:
                if (tl_x[ci] >= stl_x - 1e-6 and tl_y[ci] >= stl_y - 1e-6 and
                    br_x[ci] <= sbr_x + 1e-6 and br_y[ci] <= sbr_y + 1e-6):
                    rows.append(ii)
                    cols.append(j)
            else:
                if (tl_x[ci] >= ftl_x - 1e-6 and tl_y[ci] >= ftl_y - 1e-6 and
                    br_x[ci] <= fbr_x + 1e-6 and br_y[ci] <= fbr_y + 1e-6):
                    rows.append(ii)
                    cols.append(j)

    if not rows:
        return None

    data = np.ones(len(rows), dtype=np.float64)
    A = csc_matrix((data, (rows, cols)), shape=(n_norm, n_cand))

    uncovered = [ii for ii in range(n_norm) if A[ii].nnz == 0]
    if uncovered:
        return None

    c = np.ones(n_cand)
    constraints = LinearConstraint(A, lb=1.0)
    bounds = Bounds(0, 1)
    integrality = np.ones(n_cand)

    result = milp(c, constraints=constraints, bounds=bounds,
                  integrality=integrality,
                  options={'time_limit': 10.0})
    if not result.success:
        return None

    selected = np.where(result.x > 0.5)[0]

    fovs = []
    assigned = set()
    for j in selected:
        cx, cy, step = candidates[j]
        comp_list = []
        col_rows = A[:, j].nonzero()[0]
        for ii in col_rows:
            ci = normal_comps[ii]
            if ci not in assigned:
                comp_list.append(ci)
                assigned.add(ci)
        if comp_list:
            fovs.append((round(cx, 2), round(cy, 2), comp_list, 0))

    if len(assigned) != n_norm:
        return None

    return fovs
