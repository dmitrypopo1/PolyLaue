# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 21:28:33 2024

@author: dpopov
"""

# Copyright © 2024, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: Extended version of software PolyLaue
# By: Argonne National Laboratory, KITWARE, INC.
# BSD OPEN SOURCE LICENSE
# The full text of the license is available in file LICENSE

import numpy as np

ImageSizeX=2048 # X-ray image size along X, pixels
ImageSizeY=2048 # X-ray image size along Y, pixels

WhiteBeamShift=0.01 # Shift of white beam from mono beam 
# at the sample position along HPCAT Y+, mm

EnergyHighest=90 # Highest limit of X-ray energy, keV



# Dioptas 0.8.5
def geo_from_Dioptas():
    import math
    global ImageSizeX, ImageSizeY, WhiteBeamShift
    imsiy=float(ImageSizeY)
    wmbs=float(WhiteBeamShift)
    with open('poly.poni') as f:
        poly_poni=f.read()
    dt=[1,1,1,1,1,1]
    for w in poly_poni.splitlines():
        ww=w.split()
        if 'PixelSize1:' in ww:
            dt[0]=0
            pix=float(ww[1])*1000.0
            print('Pixel size, mm:', pix)
        if '"pixel2":' in ww:
            dt[0]=0
            ss=ww[4]
            sss=len(ww[4])
            pix=float(ss[:(sss-1)])*1000.0
            print('Pixel size, mm:', pix)
        if 'Distance:' in ww:
            dt[1]=0
            sam_det_d=float(ww[1])*1000.0
            print('Sample to detector distance, mm:', sam_det_d)
        if 'Poni1:' in ww:
            dt[2]=0
            PoniY=float(ww[1])
        if 'Poni2:' in ww:
            dt[3]=0
            PoniX=float(ww[1])
        if 'Rot1:' in ww:
            dt[4]=0
            rot1=float(ww[1])
            print('Rot1, rad:', rot1)
        if 'Rot2:' in ww:
            dt[5]=0
            rot2=float(ww[1])
            print('Rot2, rad:', rot2)
    if 1 in dt:
        return '...Error! Unknown format'
    PoniX=PoniX*1000.0/pix-wmbs/pix
    PoniY=imsiy-PoniY*1000.0/pix
    print('PoniX, pix:', PoniX)
    print('PoniY, pix:', PoniY)
    beam_x=math.sin(math.pi/2.0+rot2)*math.cos(math.pi/2.0+rot1)
    beam_y=math.cos(math.pi/2.0+rot2)
    beam_z=math.sin(math.pi/2.0+rot2)*math.sin(math.pi/2.0+rot1)
    dt=[]
    dt.append(PoniX)
    dt.append(PoniY)
    det_org=np.array(dt, dtype=np.float64)
    dt=[]
    dt.append(beam_x)
    dt.append(beam_y)
    dt.append(beam_z)
    beam_dir=np.array(dt, dtype=np.float64)
    dt=[]
    dtl=[]
    dtl.append(0)
    dtl.append(0)
    dt.append(dtl)
    dtl=[]
    dtl.append(ImageSizeX)
    dtl.append(0)
    dt.append(dtl)
    dtl=[]
    dtl.append(ImageSizeX)
    dtl.append(ImageSizeY)
    dt.append(dtl)
    dtl=[]
    dtl.append(0)
    dtl.append(ImageSizeY)
    dt.append(dtl)
    im_corn=np.array(dt, dtype=np.float64)
    dt=[]
    for i in im_corn:
        dt.append(sam_det_d)
    ang_vec1=np.hstack((((im_corn-det_org)*np.float64(pix)), np.expand_dims(np.array(dt, dtype=np.float64), axis=1)))
    ang_vec2=ang_vec1/np.expand_dims(np.sqrt(np.sum(np.square(ang_vec1), axis=1)), axis=1)
    ang_tet=math.acos(float(np.min(ang_vec2 @ beam_dir)))/2.0
    print('Largest teta, deg.:', round((ang_tet*180.0/math.pi), 2))
    ang_sol=float(np.min(ang_vec2[:,2]))
    dt=[]
    dt.append(pix)
    dt.append(sam_det_d)
    dt.append(ang_tet)
    dt.append(ang_sol)
    pix_dist=np.array(dt, dtype=np.float64)
    np.savez('geosetup.npz', iitt1=det_org, iitt2=beam_dir, iitt3=pix_dist)
    print('  ')
    print('...LaueGo detector geometry')
    Px=(float(ImageSizeX)/2.0-PoniX)*pix
    Py=(float(ImageSizeY)/2.0-PoniY)*pix
    Pz=sam_det_d
    print('Px=, Py=, Pz=', round(Px, 3), round(Py, 3), round(Pz, 3))
    Rx=beam_y
    Ry=-beam_x
    Rz=0.0
    Rr=math.sqrt(Rx*Rx+Ry*Ry)
    if Rr > 0.0003:
        R=math.acos(beam_z)
        Rx=Rx*R/Rr
        Ry=Ry*R/Rr
    else:
        Rx=0.0
        Ry=0.0
    print('Rx=, Ry=, Rz=', round(Rx, 6), round(Ry, 6), round(Rz, 6))
    
