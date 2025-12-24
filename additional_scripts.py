# -*- coding: utf-8 -*-
"""
Created on Sun Dec 21 20:35:58 2025

@author: dpopov
"""

import numpy as np

# uvw_max - the largest possible u,v,w indices of zone axes

# ang_max - the largest angle of zone axes with respect to the incident beam

def beam_zone(uvw_max=1, ang_max=30):
    import math
    with np.load('geosetup.npz') as geo_arr:
        beam_dir=geo_arr['iitt2']
    abc=np.load('abc_matrix0.npy')
    abc_dir=np.reshape(abc, (3,-1))
    mm=np.int64(uvw_max)
    sh1=np.expand_dims(np.arange(-mm, (mm+1), dtype=np.int64), axis=1)
    sk1=np.expand_dims(np.arange(-mm, (mm+1), dtype=np.int64), axis=1)
    sl1=np.expand_dims(np.arange(-mm, (mm+1), dtype=np.int64), axis=1)
    sh0=np.expand_dims(np.zeros((mm*np.int64(2)+np.int64(1)), dtype=np.int64), axis=1)
    sk0=np.expand_dims(np.zeros((mm*np.int64(2)+np.int64(1)), dtype=np.int64), axis=1)
    sl0=np.expand_dims(np.zeros((mm*np.int64(2)+np.int64(1)), dtype=np.int64), axis=1)
    sh=np.hstack((sh1,sh0,sh0))
    sk=np.hstack((sk0,sk1,sk0))
    sl=np.hstack((sl0,sl0,sl1))
    shkl=np.expand_dims((np.expand_dims(sh, axis=1)+sk), axis=2)+sl
    shkl=np.reshape(shkl, (((mm*np.int64(2)+np.int64(1))*(mm*np.int64(2)+np.int64(1))*(mm*np.int64(2)+np.int64(1))), 3))
    vec_sel=np.sum(np.absolute(shkl), axis=1) != 0
    shkl=shkl[vec_sel,:]
    shkl_vec1=shkl.astype(np.float64)
    shkl_vec1=shkl_vec1 @ abc_dir
    shkl_vec1=shkl_vec1/np.expand_dims(np.sqrt(np.sum(np.square(shkl_vec1), axis=1)), axis=1)
    ang=np.sum((shkl_vec1*beam_dir), axis=1)
    vec_sel=np.nonzero(ang > np.float64(math.cos(float(ang_max)*math.pi/180.0)))
    ang=ang[vec_sel[0]]
    shkl=shkl[vec_sel[0],:]
    ii=np.int64(0)
    print('Zone axes, angles from the incident beam')
    for aa in shkl:
        mang=math.acos(float(ang[ii]))*180.0/math.pi
        print(aa, ' , ', round(mang, 4), 'degrees')
        ii=ii+np.int64(1)
    