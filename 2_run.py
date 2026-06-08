import os
import argparse
import logging
import pandas as pd
logging.disable(logging.CRITICAL)

from utils.checker import Checker

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run solver and/or evaluate results for all PCBs.')
    parser.add_argument('--data', default='simulation_data', help='Path to the data folder containing PCB subdirectories (default: simulation_data)')
    parser.add_argument('--eval', action='store_true', help='Evaluation only mode: skip solving and evaluate existing output files')
    parser.add_argument('--solver', choices=['ky', 'greedy'], default='ky', help='Solver to use: ky (ART+MILP+beam, default) or greedy')
    args = parser.parse_args()

    if not args.eval:
        if args.solver == 'ky':
            from ky_solver._solver import solve
        else:
            from solver import solve

    B = '\033[94m'; R = '\033[91m'; G = '\033[92m'; D = '\033[90m'; RST = '\033[0m'

    root = args.data
    folders = sorted([f for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))])
    W = max(len(f) for f in folders) if folders else 10
    sep = D + '-' * (W + 48) + RST

    print(f'{B}{"PCB":>{W}} | {"FOVs":>5} | {"Core":>5} | {"Cover":>6} | {"Order":>6} | {"CT (s)":>9}{RST}')
    print(sep)

    total_ct = 0.0
    n_penalty = 0

    for folder in folders:
        path = os.path.join(root, folder)
        n = solve(path) if not args.eval else len(pd.read_csv(os.path.join(path, 'output_fov.csv')))
        c = Checker(path)
        cover = c.cover_all_components()
        order = c.fov_inspection_order()
        penalty = not cover or not order
        if penalty:
            c.penalty_solver()
            n_penalty += 1
            cover_after = not any(c.component.get('error', 0))
            order_after = all(c.fov['type'].iloc[i] >= c.fov['type'].iloc[i+1] for i in range(len(c.fov)-1))
        else:
            cover_after = cover
            order_after = order
        ct = c.save_combined_view()
        total_ct += ct
        core = getattr(c, '_optimal_core', int(pd.read_csv(os.path.join(path, 'input_parameter.csv'))['max_core'].iloc[0]))
        p = '*' if penalty else ''
        cov_color = R if not cover_after else G
        ct_color = R if penalty else ''
        ct_rst = RST if penalty else ''
        print(f'{folder:>{W}} | {n:>5} | {core:>5d} | {cov_color}{str(cover_after):>6}{RST} | {str(order_after):>6} | {ct_color}{ct:>8.2f}{p}{ct_rst}')

    print(sep)
    print(f'{"AVG":>{W}} | {"":>5} | {"":>5} | {"":>6} | {"":>6} | {total_ct / len(folders):>8.2f}')
    if n_penalty > 0:
        print(f'\n{R}* = penalty_solver applied ({n_penalty}/{len(folders)} datasets){RST}')
