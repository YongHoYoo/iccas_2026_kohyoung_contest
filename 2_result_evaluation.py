import os
import json 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from utils.checker import Checker 

if __name__ == "__main__":

    root_folder = 'example_data' 

    for folder in os.listdir(root_folder): 
        data_folder = os.path.join(root_folder, folder) 
        checker = Checker(data_folder) 
        checker.cover_all_components() 
        checker.fov_inspection_order() 
        checker.save_pcb_view() 
        checker.save_timelog() 

