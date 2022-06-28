#!/usr/bin/env python3
# coding: utf-8

# In[1]:

from __future__ import print_function

import numpy as np
from pylab import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits as fits
from astropy.wcs import WCS
from astropy.table import Table
from scipy.interpolate import griddata
from photutils.aperture import *
from scipy.ndimage import gaussian_filter
from scipy.spatial import *
from sklearn.neighbors import KDTree
from astropy.table import Table
from astropy.table import vstack as vstack_table

from skimage.restoration import denoise_tv_bregman
from scipy.signal import convolve2d
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull
import itertools


from astropy.modeling.models import Gaussian2D
from skimage import color, data, restoration
from mpl_toolkits.axes_grid1 import make_axes_locatable


import re
from astropy.nddata import Cutout2D
from reproject import reproject_interp
from collections import Counter
from image_registration import chi2_shift
from my_utils import *
#import naturalneighbor
#import _ngl
#from photutils.aperture import aperture_photometry
from matplotlib.colors import LogNorm
from matplotlib.colors import PowerNorm
#from matplotlib.nxutils import points_inside_poly
from matplotlib.path import Path as mpl_path

import Py3D
from Py3D.functions.rssMethod import *
from Py3D.functions.cubeMethod import *
from scipy.ndimage import shift as shift2D


import pickle
import pyregion
from astropy import coordinates as crd
import time

from my_gaia import *

import sfdmap


import panstarrs as PS
import wget

import os

out_dir="out"
a_nx=[]
a_ny=[]
nz=0
nx=160
ny=150

filter_r='/home/sanchez/filters/r.dat'
passband_r = PassBand()
passband_r.loadTxtFile(filter_r, wave_col=1,  trans_col=2)
eff_wave_r=passband_r.effectiveWave()
print(eff_wave_r)


list_obj=['2','NGC0171','IC0208','IC0522','IC4566',\
          'LSBCF560-04','NGC0155','NGC0529','NGC0741','NGC0932','NGC1060','NGC1070',\
          'NGC1349','NGC2253','NGC2880','NGC3158','NGC3610','NGC3614','NGC3619','NGC3893','NGC3945','NGC4047',\
          'NGC4185','NGC5406','NGC5485','NGC5576','NGC5614','NGC6125','NGC7619']

for file in sorted(os.listdir(out_dir)):
    if file.endswith(".V500.drscube.fits.gz"):
        name=file
        name = name.replace(".V500.drscube.fits.gz","")        
        if (name in list_obj):
            nz=nz+1;

print(f'{nz} files found in directory')

flat_cube=np.ones((nz,ny,nx))
r_cube=0.0001*np.ones((nz,ny,nx))

#nz=100

iz=0
for name in list_obj:
    file=name+".V500.drscube.fits.gz"
    print(file)
    if file.endswith(".V500.drscube.fits.gz"):
        if (iz<nz):
            file_now=out_dir+"/"+file
            hdu=fits.open(file_now)
            (ny_now,nx_now)=hdu[4].data.shape
            if (nx_now>nx):
                nx_now=nx
            if (ny_now>ny):
                ny_now=ny
            try:
                flux_scale=hdu[0].header['HIERARCH PIPE RAT_PS']
            except:
                flux_scale=np.nan
            flux_now=flux_scale
            if ((flux_scale>10)|(flux_scale<0.1)):
                flux_now=np.nan
            flat_cube[iz,0:ny_now,0:nx_now]=hdu[4].data[0:ny_now,0:nx_now]*flux_now

            (nz_cube,ny_cube,nx_cube)=hdu[0].data.shape        
            wavelength=hdu[0].header['CRVAL3']+hdu[0].header['CDELT3']*(arange(0,nz_cube))
            cube_cl = Cube(data=hdu[0].data,wave=wavelength,header=hdu[0].header)        
            img_r = passband_r.getFluxCube(cube_cl)[0]
            r_cube[iz,0:ny_now,0:nx_now]=img_r[0:ny_now,0:nx_now]
            file_r_img = file_now.replace(".V500.drscube.fits.gz",".r.fits.gz")
            print(f'{file_r_img} written')
            primhdu = fits.PrimaryHDU(data=img_r,header=hdu[0].header)
            hdulist=fits.HDUList([primhdu])
            hdulist.writeto(file_r_img,overwrite=True)

            print(f'{file_now} added {hdu[4].data.shape} {iz}/{nz} {flux_scale}')
        iz=iz+1

#master_flat=np.average(flat_cube, axis=0, weights=r_cube)
flat_cube[(flat_cube>10)|(flat_cube<0.1)|(r_cube>1)]=np.nan
#up_flat = np.nanmean(flat_cube*r_cube, axis=0)
#down_flat = np.nanmean(r_cube,axis=0)
#master_flat=np.nanmean(flat_cube*r_cube, axis=0)/np.nanmean(r_cube,axis=0)
master_flat=np.nanmean(flat_cube, axis=0)
master_flat = np.nan_to_num(master_flat, nan=1, posinf=1, neginf=1)
#master_flat[master_flat == np.nan]=1
primhdu = fits.PrimaryHDU(data=master_flat,header=hdu[0].header)
hdulist=fits.HDUList([primhdu])
file_out=out_dir+"/master_flat.fits.gz"
hdulist.writeto(file_out,overwrite=True)


#primhdu = fits.PrimaryHDU(data=master_flat,header=hdu[0].header)
#hdulist=fits.HDUList([primhdu])
#file_out=out_dir+"/master_flat.fits.gz"
#hdulist.writeto(file_out,overwrite=True)




        
