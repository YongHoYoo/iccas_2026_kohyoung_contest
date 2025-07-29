import os
import json 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
from utils.checker import Checker 

if __name__ == "__main__":

    for folder in os.listdir('data'): 
        data_folder = os.path.join('data', folder) 
        checker = Checker(data_folder) 
        checker.cover_all_components() 
        checker.fov_inspection_order() 
        checker.save_pcb_view() 
        checker.save_timelog() 
#        save_pcb_view(data_folder) 
#        save_timelog(data_folder) 

