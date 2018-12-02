#!/usr/bin/env python
# coding: utf-8

# In[88]:


import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

def import_data():
    
    ls_x_path = 'data/large_scale/'
    ls_y_path = 'data/large_scale-optimum/'
    ld_x_path = 'data/low-dimensional/'
    ld_y_path = 'data/low-dimensional-optimum/'
    
    ls_filenames = [f for f in listdir(ls_x_path) if isfile(join(ls_x_path, f))]
    ld_filenames = [f for f in listdir(ld_x_path) if isfile(join(ld_x_path, f))]
    
    ls_X = []
    ls_Y = []
    for filename in ls_filenames :
        #read data excluding last line (wich is the answer)
        x = pd.read_csv(ls_x_path+filename,names=['weight', 'price'], delimiter = ' ', header=None,error_bad_lines=False ) #you can use skipfooter=1 too 
        ls_X.append(x)
        y =  pd.read_csv(ls_y_path+filename,names=['optimal'], delimiter = ' ', header=None)
        ls_Y.append(y.iloc[0]['optimal'])

    ld_X = []
    ld_Y = []
    for filename in ld_filenames :
        #read data excluding last line (wich is the answer)
        x = pd.read_csv(ld_x_path+filename,names=['weight', 'price'], delimiter = ' ', header=None,error_bad_lines=False ) #you can use skipfooter=1 too 
        ld_X.append(x)
        y =  pd.read_csv(ld_y_path+filename,names=['optimal'], delimiter = ' ', header=None)
        ld_Y.append(y.iloc[0]['optimal'])
        
    
    return ls_X, ls_Y, ld_X, ld_Y

