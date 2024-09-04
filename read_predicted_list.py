# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 23:12:07 2024

@author: dpopov
"""

import numpy as np

#pred_list1: h,k,l,n1,n2
#pred_list2: x,y,energy,d

with np.load('predicted_list.npz') as pred:
    pred_list1=pred['ipred_list1']
    pred_list2=pred['ipred_list2']
    print(pred_list1)
    print(pred_list2)