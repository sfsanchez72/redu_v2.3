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
from os import path
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



# In[2]:


from matplotlib import rcParams as rc
rc.update({'font.size': 20,           'font.weight': 900,           'text.usetex': True,           'path.simplify'           :   True,           'xtick.labelsize' : 20,           'ytick.labelsize' : 20,#           'xtick.major.size' : 3.5,\
#           'ytick.major.size' : 3.5,\
           'axes.linewidth'  : 2.0,\
               # Increase the tick-mark lengths (defaults are 4 and 2)
           'xtick.major.size'        :   6,\
           'ytick.major.size'        :   6,\
           'xtick.minor.size'        :   3,\
           'ytick.minor.size'        :   3,\
           'xtick.major.width'       :   1,\
           'ytick.major.width'       :   1,\
           'lines.markeredgewidth'   :   1,\
           'legend.numpoints'        :   1,\
           'xtick.minor.width'       :   1,\
           'ytick.minor.width'       :   1,\
           'legend.frameon'          :   False,\
           'legend.handletextpad'    :   0.3,\
           'font.family'    :   'serif',\
           'mathtext.fontset'        :   'stix',\
           'axes.facecolor' : "w",\
           
          })


# In[3]:


def do_kdtree_n(X):
    mytree = KDTree(X)    
    dist, ind = mytree.query(X[:1], k=3)
    print(dist,ind)
    return np.array(dist),np.array(ind)

def do_kdtree(combined_x_y_arrays,points,k=1):
    mytree = cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points,k=k)
    return np.array(dist),np.array(indexes)

def weighted_median(data, weights):
    """
    Args:
      data (list or numpy.array): data
      weights (list or numpy.array): weights
    """
    data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
    s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
    midpoint = 0.5 * sum(s_weights)
    if any(weights > midpoint):
        w_median = (data[weights == np.max(weights)])[0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx:idx+2])
        else:
            w_median = s_data[idx+1]
    return w_median


# In[4]:


from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
from scipy.spatial import KDTree
from scipy.spatial import ConvexHull
import itertools

def unique2d(a):
    x, y = a.T
    b = x + y*1.0j 
    idx = np.unique(b,return_index=True)[1]
    return a[idx] 

def areaPoly(points):
    area = 0
    nPoints = len(points)
    j = nPoints - 1
    i = 0
    for point in points:
        p1 = points[i]
        p2 = points[j]
        area += (p1[0]*p2[1])
        area -= (p1[1]*p2[0])
        j = i
        i += 1

    area /= 2
    return area

def centroidPoly(points):
    nPoints = len(points)
    x = 0
    y = 0
    j = nPoints - 1
    i = 0

    for point in points:
        p1 = points[i]
        p2 = points[j]
        f = p1[0]*p2[1] - p2[0]*p1[1]
        x += (p1[0] + p2[0])*f
        y += (p1[1] + p2[1])*f
        j = i
        i += 1

    area = areaPoly(hull_points)
    f = area*6
    return [x/f, y/f]

def inter_map_IDW(phot_table_in,radius,crpix,cdelt,dim,                  mode='inverseDistance', sigma=1.0, radius_limit=5,                   resolution=1.0, min_fibers=3, slope=2.0, bad_threshold=0.1):
    phot_table=phot_table_in.copy()
    img = np.zeros([dim[1],dim[0]],dtype=np.float32)
    weights = np.zeros([dim[1],dim[0]], dtype=np.float32)
    xi = crpix[0]+cdelt[0]*np.arange(0,dim[0])
    yi = crpix[1]+cdelt[1]*np.arange(0,dim[1])
    xi,yi = np.meshgrid(xi,yi)
    phot_table['aperture_sum']=phot_table['aperture_sum']/(np.pi*radius**2)
    for phot_now in phot_table:
        if (phot_now['aperture_sum']>0):
            try:
                dist = np.sqrt((phot_now['xcenter']-xi)**2+(phot_now['ycenter']-yi)**2)
            except:
                dist = np.sqrt((phot_now['xcenter'].value-xi)**2+(phot_now['ycenter'].value-yi)**2)
            weights_0= np.exp(-0.5*np.power(dist/sigma,slope))
            weights_0[dist>radius_limit]=0
            img_0 = weights_0*phot_now['aperture_sum']#/(np.pi*radius_now**2)
            img += img_0
            weights += weights_0
            #print(phot_now['aperture_sum'],np.max(img_0),np.max(img))

    mask = (weights>0)  
    img[mask]=img[mask]/weights[mask]
#    img=gaussian_filter(img, sigma=0.25*resolution)
    return img,weights

def inter_cube_IDW(pos_table_x,pos_table_y,RSS,radius,crpix,cdelt,dim,                  mode='inverseDistance', sigma=1.0, radius_limit=5,                   resolution=1.0, min_fibers=3, slope=2.0, bad_threshold=0.1,):
    cube = np.zeros([dim[2],dim[1],dim[0]],dtype=np.float32)
    weights = np.zeros([dim[1],dim[0]], dtype=np.float32)
    xi = crpix[0]+cdelt[0]*np.arange(0,dim[0])
    yi = crpix[1]+cdelt[1]*np.arange(0,dim[1])
    xi,yi = np.meshgrid(xi,yi)
    RSS=RSS/(np.pi*radius**2) # Flux is now in units of spaxel    
    for i,(pos_x,pos_y) in enumerate(zip(pos_table_x,pos_table_y)):
        if (np.nanmean(RSS)>0):
            dist = np.sqrt((pos_x-xi)**2+(pos_y-yi)**2)
            weights_0= np.exp(-0.5*np.power(dist/sigma,slope))
            weights_0[dist>radius_limit]=0
            cube_0 = np.ones([dim[2],dim[1],dim[0]],dtype=np.float32)*weights_0
            #print(cube_0.shape)
            cube_0_T = cube_0.T*RSS[i,:]
            #print(cube_0_T.shape)
            cube_0=cube_0_T.transpose(2,1,0)
            #print(cube_0.shape)
            cube += cube_0
            weights += weights_0
            print(f'{i}', end="\r")
            #print(phot_now['aperture_sum'],np.max(img_0),np.max(img))
    cube_weights=np.ones([dim[2],dim[1],dim[0]],dtype=np.float32)
    cube_weights=cube_weights*weights
    mask = (cube_weights>0)  
    cube[mask]=cube[mask]/cube_weights[mask]
#    img=gaussian_filter(img, sigma=0.25*resolution)
    return cube,weights
    
def inter_map(phot_table_in,radius,crpix,cdelt,dim,method='cubic'):
    phot_table=phot_table_in.copy()
    xi = crpix[0]+cdelt[0]*np.arange(0,dim[0])
    yi = crpix[1]+cdelt[1]*np.arange(0,dim[1])
    xi,yi = np.meshgrid(xi,yi)
    phot_table['aperture_sum']=phot_table['aperture_sum']/(np.pi*radius**2)
    try:
        img = griddata((phot_table['xcenter'], phot_table['ycenter']), phot_table['aperture_sum'], (xi, yi),method=method)
    except:
        img = griddata((phot_table['xcenter'].value, phot_table['ycenter'].value), phot_table['aperture_sum'], (xi, yi),method=method)
    img=gaussian_filter(img, sigma=0.25)
    return img


#  linear_interp = scipy.interpolate.griddata(known_points, known_values, tuple(grid), method='linear')

#    nn_interp = naturalneighbor.griddata(known_points, known_values, grid_ranges)




def bilinear_interpolation(x,y,x_,y_,val):

    a = 1 /((x_[1] - x_[0]) * (y_[1] - y_[0]))
    xx = np.array([[x_[1]-x],[x-x_[0]]],dtype='float32')
    f = np.array(val).reshape(2,2)
    yy = np.array([[y_[1]-y],[y-y_[0]]],dtype='float32')
    b = np.matmul(f,yy)

    return a * np.matmul(xx.T, b)       



def Gaussian2D_map(nx,ny,A,xc,yc,sigma_x,sigma_y,theta=0):
    xi = np.arange(0,nx)
    yi = np.arange(0,ny)
    xi,yi = np.meshgrid(xi,yi)    
    img = Gaussian2D(A, xc, yc, sigma_x, sigma_y, theta=theta)(xi, yi)
    return img

def plot_three(img_in,img_now,res):
    fig,ax = plt.subplots(1,3,figsize=(10,8))
    #plt.subplot(ax[0],projection=wcs)
    ax[0].imshow(np.log10(img_in), cmap='magma_r', vmin=-3, vmax=-0.5, origin='lower')
    #plt.subplot(ax[1],projection=wcs)
    ax[1].imshow(np.log10(img_now), cmap='magma_r', vmin=-3, vmax=-0.5, origin='lower')
    #plt.subplot(ax[2],projection=wcs)
    std=np.nanstd(res)
    im=ax[2].imshow(res/img_in, cmap='coolwarm', vmin=-1, vmax=1, origin='lower')


#    ax[0].set_xlim([0.2*nx,0.8*nx])
#    ax[0].set_ylim([0.2*ny,0.8*ny])

#    ax[1].set_xlim([0.2*nx,0.8*nx])
#    ax[1].set_ylim([0.2*ny,0.8*ny])

#    ax[2].set_xlim([0.2*nx,0.8*nx])
#    ax[2].set_ylim([0.2*ny,0.8*ny])
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im,cax=cax)
#    plt.tight_layout()
    plt.show()


def get_PS_warp(name,ra,dec,size=480):
#    name = data_now['name']
#    ra = data_now['CRVAL1_CR']
#    dec = data_now['CRVAL2_CR']
#    size = 480
    fitsurls = PS.geturl(ra, dec, size=size, filters="grizy", format="fits")
    filters = ['y','z','i','r','g']
    for ind,fitsurl in enumerate(fitsurls):
        output=f'{name}_PS_warp_{filters[ind]}.fits'
        if (path.exists(output)):
            print(f'{output} exists...')
        else:
            wget.download(fitsurl,output)
            print('f{output} download...')
    colorim=PS.getcolorim(ra, dec, size=size,format='png')
    finput=f'{name}_PS_warp_g.fits'
    output=f'{name}_PS_warp.png'
    print(finput)
    fh=fits.open(finput)
    wcs = WCS(fh[0].header)
    fig,ax = plt.subplots(figsize=(7,7))
    ax=plt.subplot(projection=wcs)
    ax.imshow(colorim,origin="lower")
    fig.savefig(output,dpi=150)
    

    
def get_FWHM_conv(img_test,img_ref,FWHM_ref_arc=1,pix_size=1,plot=0):
    diff_min=1e12
    sigma_min=0.1
    for sigma in arange(0.2,10,0.1):
        s_img = gaussian_filter(hdu.data,sigma=sigma)
        diff = img_test-s_img
        d=img_test.shape
        mean_diff = np.sum(np.abs(diff[img_test>0]/img_test[img_test>0])**2)/(d[0]*d[1])
        if (mean_diff<diff_min):
            diff_min=mean_diff
            sigma_min=sigma
    FWHM_PS = FWHM_ref_arc
    FWHM_final = np.sqrt(FWHM_PS**2+sigma_min*pix_size*2.354)
    print('sigma_min_pix=',sigma_min,', sigma_min_arc=',sigma_min*pix_size)
    print('Final FWHM [arcsec] = ', FWHM_final)
    if (plot==1):
        s_img = gaussian_filter(hdu.data,sigma=sigma_min)
        plot_three(s_img,img_test,s_img-img_test)
    return FWHM_final,sigma_min

def deconv_img(img_center,aper_center,FWHM_input=1,slope=0.75,smooth=0,clip=True,filter_epsilon=0.005):
    sigma_FWHM=FWHM_input/pix_size/2.354
    (ny,nx)=img_center.shape
    img_Gauss=Gaussian2D_map(nx,ny,1,nx*0.5,ny*0.5,sigma_FWHM,sigma_FWHM,0)
    phot_Gauss = aperture_photometry(img_Gauss, aper_center,                                           method='subpixel',subpixels=5)
    img_Gauss_r,weights_Gauss_r = inter_map_IDW(phot_Gauss,radius,[0,0],[1,1],[nx,ny],slope=slope,                                                radius_limit=0.5*5/pix_size)
    img_psf_r = img_Gauss_r[int(0.4*ny):int(0.6*ny),int(0.4*nx):int(0.6*nx)]
    img_loop_in = img_center+img_center.max() * 1E-5 * np.random.random(img_center.shape)
    img_loop_dec = restoration.richardson_lucy(img_loop_in, img_psf_r,10,                                               clip=clip,filter_epsilon=filter_epsilon)#, num_iter=30)
    if (smooth==1):
        img_loop_dec = gaussian_filter(img_loop_dec,sigma=FWHM_input/pix_size/2.354)#*np.sum(img_Gauss)
    flux_ratio = np.sum(img_center)/np.sum(img_loop_dec)
    img_loop_dec = img_loop_dec*flux_ratio
    img_loop_dec[img_center==0]=0
    return img_loop_dec

def deconv_get_psf(img_center,aper_center,FWHM_input=1):
    (ny,nx)=img_center.shape
    img_Gauss=Gaussian2D_map(nx,ny,1,nx*0.5,ny*0.5,sigma_FWHM,sigma_FWHM,0)
    phot_Gauss = aperture_photometry(img_Gauss, aper_center,                                           method='subpixel',subpixels=5)
    img_Gauss_r,weights_Gauss_r = inter_map_IDW(phot_Gauss,radius,[0,0],[1,1],[nx,ny],slope=slope,                                                radius_limit=0.5*5/pix_size)
    img_psf_r = img_Gauss_r[int(0.4*ny):int(0.6*ny),int(0.4*nx):int(0.6*nx)]
    return img_psf_r

def deconv_img_psf(img_center,img_psf,FWHM_input=1,smooth=0,clip=True,filter_epsilon=0.005):
    sigma_FWHM=FWHM_input/pix_size/2.354
    (ny,nx)=img_center.shape
    img_loop_in = img_center+img_center.max() * 1E-5 * np.random.random(img_center.shape)
    img_loop_dec = restoration.richardson_lucy(img_loop_in, img_psf_r,10,                                               clip=clip,filter_epsilon=filter_epsilon)#, num_iter=30)
    if (smooth==1):
        img_loop_dec = gaussian_filter(img_loop_dec,sigma=FWHM_input/pix_size/2.354)#*np.sum(img_Gauss)
    flux_ratio = np.sum(img_center)/np.sum(img_loop_dec)
    img_loop_dec = img_loop_dec*flux_ratio
    img_loop_dec[img_center==0]=0
    return img_loop_dec


# In[5]:


def get_astrometry(img,filename_PS,name='NGC5947', ext='pdf', xc_PS=248.526778723309, yc_PS=249.418413876343, RA_PS=232.652436800918, DEC_PS=42.7171665852056, xc_cal=42.5, yc_cal=35.5, band='r',i_range=5, plot=1,new_pix=0.5,label='',FWHM_in=1,A_R=0.0, flux_lim=0.02, fname=None):
#    filename_PS = f'{name}_PS_warp_{band}.fits'
#    filename_cal = f'out/map.r.{name}.fits'
#    print(filename_PS)
    f_cor=1.0
    ww = {'g': 4866, 'r': 6215, 'i': 7545, 'z': 8679, 'y': 9633}
    ww_now = ww[band]
    if (path.exists(filename_PS)):
        if (fname==None):
            fname=f'astro_{name}_{band}_new.{ext}'
        hdu_PS = fits.open(filename_PS)[0]
        max_PS = np.nanmax(hdu_PS.data)
        print(f'max_PS = {max_PS}')
        hdu_PS.data=np.nan_to_num(hdu_PS.data,nan=max_PS)
        hdu_PS_org=hdu_PS.copy()
        (ny_PS,nx_PS)=hdu_PS.data.shape
        EXPTIME = hdu_PS.header['EXPTIME']
        wcs_copy = WCS(hdu_PS.header)
        print(f'CDELT_ORG = {wcs_copy.wcs.cdelt}')
        pix_rat = np.abs(wcs_copy.wcs.cdelt[0]*3600)
        f_ratio = ((10**(-6.64))/EXPTIME)*(((1e-23)*(3e18))/ww_now**2)*1e16/pix_rat**2
        (ny_PS,nx_PS)=hdu_PS.data.shape
        xc_PS = nx_PS/2.0
        yc_PS = ny_PS/2.0 
        hdu_PS.data=hdu_PS.data*10**(0.4*A_R)
        hdu_PS_org.data=hdu_PS_org.data*10**(0.4*A_R)
        print('xc_PS=',xc_PS,yc_PS, int(1.2*xc_PS),int(1.2*yc_PS))
        img_cut_PS =  Cutout2D(hdu_PS.data.astype(np.float32),                               (int(xc_PS),int(yc_PS)), (int(1.2*xc_PS),int(1.2*yc_PS)),wcs=wcs_copy)
        w=img_cut_PS.wcs
        header = w.to_header()
        hdu_PS = fits.PrimaryHDU(header=header)
        hdu_PS.data=img_cut_PS.data
        wcs_copy = WCS(hdu_PS.header)
        mag_PS = -2.5*np.log10(np.nansum(hdu_PS.data))+25+2.5*np.log10(EXPTIME)
        hdu_PS.data=hdu_PS.data*f_ratio
        hdu_PS_org.data=hdu_PS_org.data*f_ratio
#        plt.imshow(np.log10(img))
#        plt.show()
        
        print(f'flux_ratio = {f_ratio}')
        (ny_PS,nx_PS)=hdu_PS.data.shape
        wcs = WCS(hdu_PS.header)

        
        hdu = fits.PrimaryHDU(img)
        xc_cal=xc_cal-0.5
        yc_cal=yc_cal-0.5
        x_list = np.arange(0,nx_PS)
        y_list = np.arange(0,ny_PS)
        xx_list, yy_list = np.meshgrid(x_list, y_list)
        map_dist_PS = np.sqrt((xx_list-xc_PS)**2+(yy_list-yc_PS)**2)
        hdu_dist_PS = fits.PrimaryHDU(header=header)
        hdu_dist_PS.data=map_dist_PS

        (ny_cal,nx_cal)=hdu.data.shape
        hdu.data=hdu.data/(new_pix**2)/f_cor
        x_list = np.arange(0,nx_cal)
        y_list = np.arange(0,ny_cal)
        xx_list, yy_list = np.meshgrid(x_list, y_list)
        map_dist_CAL = np.sqrt((xx_list-nx_cal/2)**2+(yy_list-ny_cal/2)**2)
        crpix1_cal=np.int(xc_cal)+1
        crpix2_cal=np.int(yc_cal)+1
        d_crval1=new_pix*((xc_cal+1)-crpix1_cal)/3600.0
        d_crval2=new_pix*((yc_cal+1)-crpix2_cal)/3600.0
        crval1_cal=RA_PS+d_crval1
        crval2_cal=DEC_PS-d_crval2
        flux_CAL = np.nansum(hdu.data)
        flux_val = (((np.nansum(hdu.data[hdu.data>0]))*(ww_now**2))/((1e-23)*(3e18)))*1e-16
        mag_CAL = 8.4-2.5*np.log10(flux_val*(new_pix**2))
        #
        # Update the header with the new wcs!
        #
        (nx,ny)=hdu.data.shape
        wcs_cal = WCS(naxis=2)
        print('WCS_input=',crpix1_cal,crpix2_cal)
        wcs_cal.wcs.crpix = [crpix1_cal,crpix2_cal]
        wcs_cal.wcs.cdelt = np.array([-new_pix/3600,new_pix/3600])
        wcs_cal.wcs.crval = [crval1_cal, crval2_cal]
        wcs_cal.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        hdu.header=wcs_cal.to_header()
        #
        #
        #        
        img_PS_to_CAL, footprint_PS_to_CAL = reproject_interp(hdu_PS, hdu.header)
#        plt.imshow(np.log10(img_PS_to_CAL.data))
#        plt.show()
#        plt.imshow(np.log10(hdu.data))
#        plt.show()
        mag_PS_to_CAL = -2.5*np.log10(np.nansum(img_PS_to_CAL/f_ratio/(pix_rat**2)*(new_pix**2)))+25+2.5*np.log10(EXPTIME)
        print("IN= ",crval1_cal,crval2_cal)
        d_shift=0.3        
        for i in range(i_range):
            #img_PS_to_CAL, footprint_PS_to_CAL = reproject_interp(hdu_PS, hdu.header)
            img_PS_to_CAL_input, footprint_PS_to_CAL = reproject_interp(hdu_PS, hdu.header)
            img_PS_to_CAL=gaussian_filter(img_PS_to_CAL_input, sigma=1.0)
            xoff, yoff, exoff, eyoff = chi2_shift(img_PS_to_CAL, hdu.data, err=map_dist_CAL/30,
                                                  return_error=True, upsample_factor='auto')#,boundary='wrap')
            if ((np.abs(xoff)<10) and (np.abs(yoff)<10) and (np.abs(eyoff)<10) and (np.abs(eyoff)<10)):
#                print('Offsets (',i,'):  x= ',xoff,'+-',exoff,' y= ',yoff,'+-',eyoff)
                crval1_cal=crval1_cal-xoff*(-new_pix/3600)
                crval2_cal=crval2_cal-yoff*(new_pix/3600)
                e_crval1_cal=exoff*(-new_pix/3600)
                e_crval2_cal=eyoff*(new_pix/3600)
                wcs_cal.wcs.crval = [crval1_cal, crval2_cal]
                hdu.header=wcs_cal.to_header()
            else:
                e_crval1_cal=exoff*(-new_pix/3600)
                e_crval2_cal=eyoff*(new_pix/3600)
        print("OUT = ",crval1_cal,crval2_cal)
        if (filename_PS.find("_PS_")):
            print('photometric correction')
            pix_rat = np.abs(wcs_copy.wcs.cdelt[0]/(new_pix/3600))
            print(f'pix_rat = {pix_rat}')
            rat_data=hdu.data/img_PS_to_CAL#(pix_rat)#/hdu.data
#            plt.imshow(hdu.data)
            mask_data= (hdu.data>0.001)
            print(len(rat_data[mask_data]))
#            rat_mean=weighted_median(rat_data[mask_data],1/(1+map_dist_CAL[mask_data]))
            ma = np.ma.MaskedArray(rat_data[mask_data], mask=np.isnan(rat_data[mask_data]))
            rat_mean=np.ma.average(ma, weights=hdu.data[mask_data])
            #rat_mean=np.nanmedian(rat_data[mask_data])
            rat_std=np.nanstd(rat_data[mask_data])
            print(f'rat = {rat_mean}+-{rat_std}')

        
        #wcs_cal = WCS(hdu.header)    
        
        print(f'filter={band}, mag_PS = {mag_PS}, mag_PS_to_CAL = {mag_PS_to_CAL}, mag_CAL = {mag_CAL}')
        mask_now = (hdu.data>flux_lim) & (img_PS_to_CAL>flux_lim)
        n_mask=mask_data.sum()
        sigma_now = 1
        chi_now=1e12
        if (n_mask>10):
            sigma_list=[]
            chi_list=[]
            for sigma in np.arange(0.1,3.3,0.05):
                img_PS_conv=gaussian_filter(hdu_PS.data, sigma=sigma/pix_rat)
                hdu_PS_conv = hdu_PS.copy()
                hdu_PS_conv.data=img_PS_conv
                img_PS_to_CAL_input, footprint_PS_to_CAL = reproject_interp(hdu_PS_conv, hdu.header)
                img_PS_to_CAL = img_PS_to_CAL_input*rat_mean                
                                
               
            
#                img_PS_to_CAL=gaussian_filter(img_PS_to_CAL_input*rat_mean, sigma=sigma)
                #chi_map = (np.sinh(img_PS_to_CAL)-np.sinh(hdu.data))**2/np.sinh(hdu.data)
                chi_map = (img_PS_to_CAL-hdu.data)**2/hdu.data
                chi_now = np.nansum(chi_map[mask_now])
                sigma_list.append(sigma)
                chi_list.append(chi_now)
            sigma_list=np.array(sigma_list)
            chi_list=np.array(chi_list)
            sigma_now=sigma_list[np.argmin(chi_list)]
            chi_now=chi_list[np.argmin(chi_list)]
        #print(sigma_list,chi_list)
        print(f'sigma_psf = {sigma_now} pixels, chi_min={chi_now}')
        #FWHM_now = np.sqrt((2.354*sigma_now*new_pix)**2+FWHM_in)
        FWHM_now = 2.354*new_pix*np.sqrt((sigma_now)**2+(FWHM_in/2.354)**2)
        FWHM_now = np.abs(FWHM_now-new_pix)
        img_PS_to_CAL_input, footprint_PS_to_CAL = reproject_interp(hdu_PS_org, hdu.header)
        img_PS_to_CAL=gaussian_filter(img_PS_to_CAL_input*rat_mean, sigma=sigma_now)
        rat_data=hdu.data/img_PS_to_CAL
        print(f'FWHM_now = {FWHM_now} arcsec')
        #------ PLOT ------ #
        vmin=-3
        vmax=0.05
        if (plot==1):
            fig=plt.figure(figsize=(15,8))
            gs = fig.add_gridspec(2,3)
            ax=fig.add_subplot(gs[0,0],projection=wcs_copy)
            ax.imshow(np.log10(hdu_PS.data), cmap='gray_r',vmin=vmin, vmax=vmax, origin='lower')
            ax.set_title('PANStarr')
            ax.set_xlabel('R.A.')
            ax.set_ylabel('Dec')
            ax.scatter(xc_PS,yc_PS, s=15,  edgecolor='blue', facecolor='blue') 
            ax3=fig.add_subplot(gs[1,0],projection=wcs_cal)
            ax3.imshow(np.log10(img_PS_to_CAL), cmap='gray_r',vmin=vmin, vmax=vmax, origin='lower')        
            ax2=fig.add_subplot(gs[0,1],projection=wcs_cal)
            ax2.imshow(np.log10(hdu.data), cmap='gray_r', vmin=vmin, vmax=vmax, origin='lower')
            ax2.contour(hdu_PS.data, levels=np.logspace(-2, 0.15, 5), colors='blue', alpha=0.5, transform=ax2.get_transform(WCS(hdu_PS.header)))
            ax2.scatter(xc_cal, yc_cal, s=20, edgecolor='white', facecolor='white')
            ax2.scatter(RA_PS, DEC_PS, s=10, transform=ax2.get_transform('fk5'),  edgecolor='blue', facecolor='blue')
            ax2.set_title('CALIFA: '+name)
            ax2.set_xlabel('R.A.')
            ax2.set_xlim(-0.5, hdu.data.shape[1] - 0.5)
            ax2.set_ylim(-0.5, hdu.data.shape[0] - 0.5)
            ax.contour(hdu.data, levels=np.logspace(-2, 0.15, 5), colors='red', alpha=0.5, transform=ax.get_transform(wcs_cal))
            ax3.contour(hdu.data, levels=np.logspace(-2, 0.15, 5), colors='red', alpha=0.5, transform=ax3.get_transform(wcs_cal))
            ax3.scatter(RA_PS, DEC_PS, s=10, transform=ax3.get_transform('fk5'),  edgecolor='blue', facecolor='blue')
            ax3.set_title('PS resampled to CALIFA: '+name)
            ax3.set_xlabel('R.A.')
            ax3.set_ylabel('Dec')
            ax.set_xlim(-0.5, hdu_PS.data.shape[1] - 0.5)
            ax.set_ylim(-0.5, hdu_PS.data.shape[0] - 0.5)
            ax.contour(hdu.data, levels=np.logspace(-2, 0.15, 5), colors='red', alpha=0.25, transform=ax.get_transform(wcs_cal))
            ax4=fig.add_subplot(gs[1,1],projection=wcs_cal)
            fig_img=ax4.imshow(np.log10(hdu.data), cmap='gray_r', vmin=vmin, vmax=vmax, origin='lower')
            ax4.contour(hdu.data, levels=np.logspace(-2, 0.15, 5), colors='red', alpha=0.5, transform=ax4.get_transform(wcs_cal),lw=3,label='CALIFA')
            ax4.contour(img_PS_to_CAL, levels=np.logspace(-2, 0.15, 5), colors='blue', alpha=0.5, transform=ax4.get_transform(wcs_cal),label='PanStarrs')
            ax4.scatter(xc_cal, yc_cal, s=20, edgecolor='white', facecolor='white')
            ax4.scatter(RA_PS, DEC_PS, s=10, transform=ax4.get_transform('fk5'),  edgecolor='blue', facecolor='blue')#,label='PanStarrs')
            ax4.set_title('WCS registration'+name)
            ax4.set_xlabel('R.A.')
            ax4.set_xlim(-0.5, hdu.data.shape[1] - 0.5)
            ax4.set_ylim(-0.5, hdu.data.shape[0] - 0.5)
            if (filename_PS.find("_PS_")):
                ax5=fig.add_subplot(gs[0,2])
                ax5.set_title(label)
                ax5.scatter(np.log10(hdu.data[mask_data]),rat_data[mask_data],                            edgecolor='none',color='black',alpha=0.2)
                ax5.set_ylim([0,2])
            ax6=fig.add_subplot(gs[1,2],projection=wcs_cal)
            fig_res=ax6.imshow(rat_data, cmap='gray_r', vmin=0.5, vmax=1.5, origin='lower')
            ax6.contour(hdu_PS.data, levels=np.linspace(1, 5, 5), colors='blue', alpha=0.5,                        transform=ax4.get_transform(WCS(hdu_PS.header)))
            ax6.scatter(xc_cal, yc_cal, s=20, edgecolor='white', facecolor='white')
            ax6.scatter(RA_PS, DEC_PS, s=10, transform=ax4.get_transform('fk5'),  edgecolor='blue', facecolor='blue')
            ax6.set_title('WCS registration'+name)
            ax6.set_xlabel('R.A.')
            ax6.set_xlim(-0.5, hdu.data.shape[1] - 0.5)
            ax6.set_ylim(-0.5, hdu.data.shape[0] - 0.5)
            fig.tight_layout()
            fig.savefig(fname,dpi=150)
#---------- END PLOT ----------#        
        return (crval1_cal,crval2_cal,crpix1_cal,crpix2_cal,e_crval1_cal,e_crval2_cal,rat_mean,rat_std,mag_PS, mag_PS_to_CAL, mag_CAL,FWHM_now,chi_now,rat_data)


# In[6]:


def register_PS_file(rss,filename_PS,band='r',verbose=1,search_box=[20.0,2.6],step=[1.0,0.2],spa=0,parallel=1,offset_x =0.0,offset_y=0.0,quality_figure='test.png'):
    # Registration
    ww = {'g': 4866, 'r': 6215, 'i': 7545, 'z': 8679, 'y': 9633}
    ww_now = ww[band]
    hdu_PS = fits.open(filename_PS)[0]
    max_PS = np.nanmax(hdu_PS.data)
    np.nan_to_num(max_PS,nan=max_PS)
    (ny_PS,nx_PS)=hdu_PS.data.shape
    EXPTIME = hdu_PS.header['EXPTIME']
    wcs_copy = WCS(hdu_PS.header)
    pix_rat = np.abs(wcs_copy.wcs.cdelt[0]*3600)
    f_ratio = ((10**(-6.64))/EXPTIME)*(((1e-23)*(3e18))/ww_now**2)*1e16/pix_rat**2
    (ny_PS,nx_PS)=hdu_PS.data.shape
    xc_PS = nx_PS/2.0
    yc_PS = ny_PS/2.0 
    pix_coordinates = [xc_PS,yc_PS]
    hdu_PS.data=hdu_PS.data*10**(0.4*Ar_name)*f_ratio/10 # No clue of where the 10.5 comes from!
    (ny,nx)=hdu_PS.data.shape    
    scale=pix_rat
    img_ref=Image(data=hdu_PS.data,error=None)
    for i in range(len(search_box)):
        if verbose==1:
            print('Start iteration %d'%(i+1))
            print('Searchbox %.2f arcsec with sampling of %.2f arcsec'%(search_box[i], step[i]))
        if i>0:
            offset_x = best_offset_x
            offset_y = best_offset_y
        (offsets_xIFU, offsets_yIFU, chisq, scale_flux, AB_flux,valid_fibers) = rss.registerImage_PS(img_ref, passband_r, search_box[i], step[i], pix_coordinates[0], pix_coordinates[1], scale, spa, offset_x, offset_y, parallel=parallel)
        idx = numpy.indices(chisq.shape)
#        print('chisq=',chisq)
        select_best = numpy.nanmin(chisq) == chisq
#        print('shape=', offsets_xIFU.shape)
#        print('offset_xIFU=', offsets_xIFU)
#        print('select_best=',select_best)
        best_offset_x = offsets_xIFU[select_best][0]
        best_offset_y = offsets_yIFU[select_best][0]
        best_chisq = chisq[select_best][0]
        best_scale = scale_flux[select_best][0]
        best_valid = valid_fibers[select_best][0]    
    
    # Plotting the results
    posTab = rss.getPositionTable()
#    flux =  img_ref.extractApertures(posTab, pix_coordinates[0], pix_coordinates[1], scale, angle=spa, offset_arc_x=best_offset_x, offset_arc_y=best_offset_y)
    flux =  img_ref.extractApertures(posTab, pix_coordinates[0]-1, pix_coordinates[1]-1, scale, angle=spa, offset_arc_x=best_offset_x, offset_arc_y=best_offset_y)

    fib_scale = AB_flux.ravel()/(flux[0]/best_scale).ravel()
#    print(flux)
#    print('#flux =',len(flux))
#    print('#AB_flux =',len(AB_flux))
#    for (f_now,AB_now) in zip(flux[0],AB_flux.ravel()):
#        print('fluxes=',f_now,AB_now)
# 
#    fig,ax = plt.subplots(1,figsize=(5,6))
#    plt.show()

  

    fig = plt.figure(figsize=(17,6))
    ax1 = fig.add_axes([0.01,0.08,0.3,0.79])
    ax2 = fig.add_axes([0.32,0.08,0.3,0.79])
    x_pos = rss._arc_position_x+best_offset_x
    y_pos = rss._arc_position_y+best_offset_y
    AB_flux[numpy.abs(AB_flux) == numpy.inf] = numpy.nan
    select_nan = numpy.isnan(AB_flux) 
    vmin = numpy.min(AB_flux[numpy.logical_not(select_nan)])
    vmax = numpy.max(AB_flux[numpy.logical_not(select_nan)])
    print('*** vmin,vmax 1 =',vmin,vmax)
    if (vmin==vmax):
        vmin=-1
        vmax=1

    norm = matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax)
    XY = numpy.hstack(((x_pos).ravel()[:, numpy.newaxis], (y_pos).ravel()[:, numpy.newaxis]))
    circ = matplotlib.collections.CircleCollection([120]*len(y_pos), offsets=XY,        transOffset=ax1.transData,norm=norm,cmap=matplotlib.cm.gist_stern_r)

    AB_flux[select_nan] = 1e-30
    circ.set_array(AB_flux.ravel())
    ax1.add_collection(circ)
    ax1.autoscale_view()
    ax1.set_xlim(-40, 40)
    ax1.set_ylim(-40, 40)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('CALIFA r band',fontsize=18,fontweight='bold')

    XY = numpy.hstack(((x_pos).ravel()[:, numpy.newaxis], (y_pos).ravel()[:, numpy.newaxis]))
    circ2 = matplotlib.collections.CircleCollection([120]*len(y_pos),offsets=XY,                             transOffset=ax2.transData,norm=norm,                                               cmap=matplotlib.cm.gist_stern_r)
    select_nan=numpy.isnan(flux[0])
    flux[0][select_nan] = 1e-30
    circ2.set_array((flux[0]/best_scale).ravel())
    ax2.add_collection(circ2)
    ax2.autoscale_view()
    ax2.set_xlim(-40, 40)
    ax2.set_ylim(-40, 40)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('PanStarrs best-match CALIFA map',fontsize=18,fontweight='bold')

    ax3 = fig.add_axes([0.66,0.08,0.35,0.83])
    chisq[numpy.abs(chisq) == numpy.inf] = numpy.nan    
    select_chisq = numpy.isnan(chisq) 
    vmin=10.0/float(np.abs(best_valid))
#    print('chisq =',chisq)
#    print('chisq_nan =',chisq[~select_chisq])
    vmax=numpy.nanmax(chisq[~select_chisq])
    print('*** vmin,vmax 2 =',vmin,vmax)
    if (vmin==vmax):
        vmin=-1
        vmax=1
    norm = matplotlib.colors.LogNorm(vmin=vmin,vmax=vmax)
    chi_map=ax3.imshow(chisq.T,origin='lower',interpolation='nearest',norm=norm,extent=[offsets_xIFU[0, 0]-step[i]/2.0,offsets_xIFU[-1, 0]+step[i]/2.0, offsets_yIFU[0, 0]-step[i]/2.0,offsets_yIFU[0, -1]+step[i]/2.0])
    
    ax3.plot(best_offset_x, best_offset_y, 'ok', ms=8)
    #cb = plt.colorbar(chi_map,ax=ax3,pad=0.0)
    ax3.set_xlabel('offset in RA [arcsec]',fontsize=18)
    ax3.set_ylabel('offset in DEC [arcsec]',fontsize=18)
    ax3.minorticks_on()
    for line in        ax3.xaxis.get_ticklines()+ax3.yaxis.get_ticklines()+ax3.xaxis.get_minorticklines()+ax3.yaxis.get_minorticklines():
        line.set_markeredgewidth(2.0)
    ax3.set_title('$\mathbf{\chi^2}$ matching for offsets',fontsize=18,fontweight='bold')
    ax3.set_xlim([offsets_xIFU[0, 0]-step[i]/2.0,offsets_xIFU[-1, 0]+step[i]/2.0])
    ax3.set_ylim([offsets_yIFU[0, 0]-step[i]/2.0,offsets_yIFU[0, -1]+step[i]/2.0])
    fig.tight_layout()
    plt.savefig(quality_figure)
#    plt.show()
                
    return (best_offset_x,best_offset_y,best_chisq,best_scale,best_valid,fib_scale,(flux[0]/best_scale).ravel())


# In[ ]:





# In[7]:


def plot_four(img_in,phot_table,img,img_loop):
    fig,ax = plt.subplots(2,2,figsize=(10,8))
    try:
        plt.subplot(ax[0][0],projection=wcs)
    except:
        print('no WCS')
    ax[0][0].imshow(np.log10(img_in), cmap='gray_r', vmin=-3, vmax=0.05, origin='lower')
    try:
        plt.subplot(ax[1][0],projection=wcs)
    except:
        print('no WCS')
    ax[0][1].imshow(np.log10(img_in), cmap='gray_r', vmin=-3, vmax=-3, origin='lower')
    cm = plt.cm.get_cmap('Greys')
    ax[0][1].scatter(phot_table['xcenter'],phot_table['ycenter'],                     c=np.log10(phot_table['aperture_sum']/(np.pi*radius**2)),                     cmap=cm, vmin=-3, vmax=0.15,                     s=140)
    try:
        plt.subplot(ax[1][0],projection=wcs)
    except:
        print('no WCS')
    ax[1][0].imshow(np.log10(img), cmap='gray_r', vmin=-3, vmax=0.05, origin='lower')
    try:
        plt.subplot(ax[1][1],projection=wcs)
    except:
        print('no WCS')
    ax[1][1].imshow(np.log10(img_loop), cmap='gray_r', vmin=-3, vmax=0.05, origin='lower')


    ax[0][0].set_xlim([0*nx,1.0*nx])
    ax[0][0].set_ylim([0*ny,1.0*ny])

    ax[0][1].set_xlim([0*nx,1.0*nx])
    ax[0][1].set_ylim([0*ny,1.0*ny])

    ax[1][0].set_xlim([0*nx,1.0*nx])
    ax[1][0].set_ylim([0*ny,1.0*ny])

    ax[1][1].set_xlim([0*nx,1.0*nx])
    ax[1][1].set_ylim([0*ny,1.0*ny])
    
    
def plot_three(img_in,img_now,res):
    fig,ax = plt.subplots(1,3,figsize=(10,8))
    #plt.subplot(ax[0],projection=wcs)
    ax[0].imshow(np.log10(img_in), cmap='gray_r', vmin=-3, vmax=0.15, origin='lower')
    #plt.subplot(ax[1],projection=wcs)
    ax[1].imshow(np.log10(img_now), cmap='gray_r', vmin=-3, vmax=0.15, origin='lower')
    #plt.subplot(ax[2],projection=wcs)
    std=np.nanstd(res)
    ax[2].imshow(res/img_in, cmap='gray_r', vmin=-2, vmax=2, origin='lower')


    ax[0].set_xlim([0.2*nx,0.8*nx])
    ax[0].set_ylim([0.2*ny,0.8*ny])

    ax[1].set_xlim([0.2*nx,0.8*nx])
    ax[1].set_ylim([0.2*ny,0.8*ny])

    ax[2].set_xlim([0.2*nx,0.8*nx])
    ax[2].set_ylim([0.2*ny,0.8*ny])
    plt.show()


# In[8]:


def DARcorr(cube,spec2_x,spec2_y,wave_norm=5500,verbose=0):
    cent_x = np.nanmean(spec2_x._data[np.abs(spec2_x._wave-wave_norm)<100])
    cent_y = np.nanmean(spec2_y._data[np.abs(spec2_y._wave-wave_norm)<100])
    off_x=cent_x-spec2_x._data
    off_y=cent_y-spec2_y._data    
    (nz,ny,nx)=cube.shape
    out_cube=np.zeros([nz,ny,nx],dtype=np.float32)
    for i in arange(nz):
        if (verbose==1):
            print(f'{i}/{nz}',end='\r')
        img = cube[i,:,:]
        img_out = shift2D(img, [off_y[i],off_x[i]], order=3, mode='constant', cval=0.0, prefilter=True)
        out_cube[i,:,:]=img_out
    return out_cube


# In[9]:

start_time = time.time()


#name = 'NGC2906'
#name = 'NGC5947'
#name = 'NGC2916'
#name = 'ARP220'

nargs=len(sys.argv)
if (nargs==2):
    name=sys.argv[1]
else:
    print('USE: create_CALIFA_cube.py NAME')
    print('Run from:')
    print('/disk-c/sanchez/CALIFA/ancillary')
    exit()

#name = 'UGC10043'
fdir = 'data'
fdir_out = 'out'
fdir_gaia = 'gaia'
fdir_galext = '/disk-c/sanchez/CALIFA/ancillary/sfddata-master/'
file_rss = f'{fdir}/{name}.mos.V500.rss.fits.gz'
file_cube = f'{fdir}/{name}.V500.rscube.fits.gz'
hdu_cube=fits.open(file_cube)
hdr_cube=hdu_cube[0].header
hdu_cube.close()
reg=hdr_cube['REGISTER']
if ((reg==1) or (reg==True)):
    already_reg=1
    print('Using previous registration')
else:
    already_reg=0
    print('Registration is required') 


# In[10]:


hdu_rss=fits.open(file_rss)
rss=loadRSS(file_rss)
rss_data=rss._data
rss_err=rss._error
rss_bad=rss._mask



#print(rss.header)


# In[11]:


#rss_data=hdu_rss[0].data
#rss_data=nan_to_num(rss_data, nan=0.0)
#plt.show()
#rss_err=hdu_rss['ERROR'].data
#rss_bad=hdu_rss['BADPIX'].data
#tab_rss=Table(hdu_rss['POSTABLE'].data)

f_e = 0.005
fsize=2.68/2
radius=fsize
(ny_rss,nx_rss)=rss_data.shape
wavelength=hdu_rss[0].header['CRVAL1']+hdu_rss[0].header['CDELT1']*(arange(0,nx_rss))


# In[12]:


#
# Refine the error map
#
med_err = np.nanmedian(rss_err)
std_err = np.nanstd(rss_err[rss_err<3*med_err])
rss_err[rss_err>(med_err+3*std_err)]=med_err


# In[13]:


#primhdu = fits.PrimaryHDU(data=rss_err_new)
#hdulist=fits.HDUList([primhdu])
#outfile=f'out/{name}.rss_err_new.fits.gz'
#hdulist.writeto(outfile,overwrite=True)


# In[ ]:





# In[14]:



get_RA_DEC="../analysis/tables/get_RA_DEC.csv"
col_RA_DEC=header_columns_space(get_RA_DEC,2)
tab_RA_DEC=ascii.read(get_RA_DEC, delimiter=',', guess=True, comment='\s*#', names=col_RA_DEC,                      fill_values=[('BAD', np.nan)])
tab_RA_DEC.rename_column('NAME', 'name')
tab_RA_DEC_name=tab_RA_DEC[tab_RA_DEC['name']==name]

get_GE="../analysis/tables/get_GALEXT.csv"
col_GE=header_columns_space(get_GE,2)
tab_GE=ascii.read(get_GE, delimiter=',', guess=True, comment='\s*#', names=col_GE,                      fill_values=[('BAD', np.nan)])
tab_GE.rename_column('NAME', 'name')
tab_GE.add_column(0.99*tab_GE['GALEXT_V'],name='Ag');
tab_GE.add_column(0.99*tab_GE['GALEXT_V']*0.8782/1.2163,name='Ar');
tab_GE.add_column(0.99*tab_GE['GALEXT_V']*0.674/1.2163,name='Ai');

tab_GE_name=tab_GE[tab_GE['name']==name]

if (len(tab_GE_name)>0):
    Ar_name = tab_GE_name['Ar'][0]
    Ag_name = tab_GE_name['Ag'][0]
else:
    print(f'No Gal extinction in the file {get_GE}')
    m = sfdmap.SFDMap(fdir_galext)
    if (reg==1):
        Ag_name=3.1*m.ebv(hdr_cube['CRVAL1'], hdr_cube['CRVAL2'])
        Ar_name = Ag_name*0.8782/1.2163
        get_PS_warp(name,hdr_cube['CRVAL1'], hdr_cube['CRVAL2'])
    else:
        print('Not registered: RA,DEC unsure!')
        hdr_cube['CRVAL1']=tab_RA_DEC_name['RA'].value[0]
        hdr_cube['CRVAL2']=tab_RA_DEC_name['DEC'].value[0]
        Ag_name=3.1*m.ebv(hdr_cube['CRVAL1'], hdr_cube['CRVAL2'])
        Ar_name = Ag_name*0.8782/1.2163
        get_PS_warp(name,hdr_cube['CRVAL1'], hdr_cube['CRVAL2'])
print('A_r =',Ar_name)
print('A_g =',Ag_name)


# In[15]:


filter_r='/home/sanchez/filters/r.dat'
passband_r = PassBand()
passband_r.loadTxtFile(filter_r, wave_col=1,  trans_col=2)
eff_wave_r=passband_r.effectiveWave()
print(eff_wave_r)
filter_g='/home/sanchez/filters/g.dat'
passband_g = PassBand()
passband_g.loadTxtFile(filter_g, wave_col=1,  trans_col=2)
eff_wave_g=passband_g.effectiveWave()
print(eff_wave_g)
filter_u='/home/sanchez/filters/u.dat'
passband_u = PassBand()
passband_u.loadTxtFile(filter_u, wave_col=1,  trans_col=2)
eff_wave_u=passband_u.effectiveWave()
print(eff_wave_u)


# In[16]:


#
# Position table from previous reduction
#already_reg=0
sel_obj1 = np.arange(0,331)
sel_obj2 = np.arange(331,662)
sel_obj3 = np.arange(662,993)
rss_obj1 = rss.subRSS(sel_obj1)
rss_obj2 = rss.subRSS(sel_obj2)
rss_obj3 = rss.subRSS(sel_obj3)

search_box=[20.0,2.5]
step=[1.5,0.125]

#search_box=[10.0,2]
#step=[0.5,0.125]

band='r'
i_range=5
filename_PS = f'{name}_PS_warp_{band}.fits'
qc_file=f'{fdir_out}/reg_{name}_obj_1.png'
(off_x_obj1,off_y_obj1,best_chisq_obj1,best_scale_obj1,best_valid_obj1,fib_scale1,flux_ref1)=register_PS_file(rss_obj1,filename_PS,band='r',verbose=1,search_box=search_box,                 step=step,spa=0,parallel=1,offset_x =0.0,offset_y=0.0,quality_figure=qc_file)
print(off_x_obj1,off_y_obj1,best_chisq_obj1,best_scale_obj1,best_valid_obj1)

qc_file=f'{fdir_out}/reg_{name}_obj_2.png'
(off_x_obj2,off_y_obj2,best_chisq_obj2,best_scale_obj2,best_valid_obj2,fib_scale2,flux_ref2)=register_PS_file(rss_obj2,filename_PS,band='r',verbose=1,search_box=search_box,                 step=step,spa=0,parallel=1,offset_x =0.0,offset_y=0.0,quality_figure=qc_file)
print(off_x_obj2,off_y_obj2,best_chisq_obj2,best_scale_obj2,best_valid_obj2)

qc_file=f'{fdir_out}/reg_{name}_obj_3.png'
(off_x_obj3,off_y_obj3,best_chisq_obj3,best_scale_obj3,best_valid_obj3,fib_scale3,flux_ref3)=register_PS_file(rss_obj3,filename_PS,band='r',verbose=1,search_box=search_box,                 step=step,spa=0,parallel=1,offset_x =0.0,offset_y=0.0,quality_figure=qc_file)
print(off_x_obj3,off_y_obj3,best_chisq_obj3,best_scale_obj3,best_valid_obj3)

rss_obj1.offsetPosTab(-1*off_x_obj1, -1*off_y_obj1)
rss_obj2.offsetPosTab(-1*off_x_obj2, -1*off_y_obj2)
rss_obj3.offsetPosTab(-1*off_x_obj2, -1*off_y_obj3)

rss_new=rss_obj1
rss_new.append(rss_obj2)
rss_new.append(rss_obj3)
PosTable_new = rss_new.getPositionTable()
pt_x=PosTable_new._arc_position_x
pt_y=PosTable_new._arc_position_y


if (already_reg==1):
    pt_file=f'{fdir}/{name}.mos.V500.pt.txt'
    with open(pt_file) as fp:
        line = fp.readline()
        pt_shape=line.split()
        pt_x=[]
        pt_y=[]
        while line:
            line = fp.readline()
            pt_pos = line.split()
            if (len(pt_pos)>0):
                pt_x.append(float(pt_pos[1]))
                pt_y.append(float(pt_pos[2]))

    pt_x=np.array(pt_x)
    pt_y=np.array(pt_y)

    
#print(pt_x,pt_y)
print(len(pt_x))
#print(pt_x)   
points=np.zeros((len(pt_x),2))
points[:,0]=pt_x
points[:,1]=pt_y


x_min = np.min(pt_x)-2*fsize
x_max = np.max(pt_x)+2*fsize
y_min = np.min(pt_y)-2*fsize
y_max = np.max(pt_y)+2*fsize
pix_size=0.5
nx = int((x_max-x_min)/pix_size)
ny = int((y_max-y_min)/pix_size)
dim = np.array([nx,ny])

pt_x_spax=(pt_x-x_min)/pix_size
pt_y_spax=(pt_y-y_min)/pix_size

points_WCS=points.copy()
points_WCS[:,0]=pt_x_spax
points_WCS[:,1]=pt_y_spax
#
# Based on the simulations
#
slope=1.25
slope_var=0.75
# Limit of S/N to consider for the PSflat
f_e = 0.005

#
#slope=0.75
sigma=pix_size
r_lim=1.5*radius/pix_size
smooth=0


# In[17]:


#fig,ax = plt.subplots(1,figsize=(4,4))
med_scale1=np.nanmedian(fib_scale1[fib_scale1>0])
med_scale2=np.nanmedian(fib_scale2[fib_scale2>0])
med_scale3=np.nanmedian(fib_scale3[fib_scale3>0])
std_scale1=np.nanstd(fib_scale1[fib_scale1>0])
std_scale2=np.nanstd(fib_scale2[fib_scale2>0])
std_scale3=np.nanstd(fib_scale3[fib_scale3>0])
print(med_scale1,std_scale1)
print(med_scale2,std_scale2)
print(med_scale3,std_scale3)
med_scale=(med_scale1+med_scale2+med_scale3)/3
s=0.1
#hist1=ax.hist(np.log10(fib_scale1),bins=50,range=(np.log10(med_scale*s),np.log10(med_scale/s)))
#hist2=ax.hist(np.log10(fib_scale2),bins=50,range=(np.log10(med_scale*s),np.log10(med_scale/s)))
#hist3=ax.hist(np.log10(fib_scale3),bins=50,range=(np.log10(med_scale*s),np.log10(med_scale/s)))

fib_scale=np.concatenate((fib_scale1,fib_scale2,fib_scale3))
flux_ref=np.concatenate((flux_ref1,flux_ref2,flux_ref3))
wslice = (wavelength > 6000) * (wavelength < 6500)
#print(wslice)
slice_r = np.nanmean(rss._data[:,wslice],1)
med_slice=np.nanmedian(slice_r)
med_flux_ref=np.nanmedian(flux_ref)
#print(len(slice_r))


#aper = CircularAperture(points_WCS, radius)
#hdu_PS = fits.open(filename_PS)[0]
#phot_PS = aperture_photometry(hdu_PS.data, aper,\
#                              method='subpixel',subpixels=5)
#print(phot_PS)


# In[18]:


mean_scale=np.nanmean([best_scale_obj1,best_scale_obj2,best_scale_obj3])
std_scale=np.nanstd([best_scale_obj1,best_scale_obj2,best_scale_obj3])
rel_scale_obj1=best_scale_obj1/mean_scale
rel_scale_obj2=best_scale_obj2/mean_scale
rel_scale_obj3=best_scale_obj3/mean_scale
std_scale=np.nanstd([rel_scale_obj1,rel_scale_obj2,rel_scale_obj3])
print(rel_scale_obj1,rel_scale_obj2,rel_scale_obj3, std_scale)


# In[19]:


#print(flux_ref[0])


# In[20]:

print('Broken fibers')
mask_broken=[]
list_broken=np.array([51,52,53,292])

#for i in arange(0,331):
for i in list_broken:
    print(f'broken[{i}] = {fib_scale[i]} vs {med_scale}')
    if ((fib_scale[i]<med_scale*s) or (fib_scale[i]>(1+s)*med_scale)):
        if (((fib_scale[i+331]<med_scale*s) or (fib_scale[i+331]>(1+s)*med_scale)) or ((fib_scale[i+662]<med_scale*s) or (fib_scale[i+662]>(1+s)*med_scale))):
#        if ((((np.log10(fib_scale[i+331])<np.log10(med_scale*s)) or (np.log10(fib_scale[i+331])>np.log10(med_scale/s)))) or (((np.log10(fib_scale[i+662])<np.log10(med_scale*s)) or (np.log10(fib_scale[i+662])>np.log10(med_scale/s))))):
            print(f'Fiber number {i} is broken')
            mask_broken.append(i)
            mask_broken.append(i+331)
            mask_broken.append(i+662)
mask_broken=np.array(mask_broken)

#
# Broken fibers
#
if (len(mask_broken)>0):
    rss_data[mask_broken,:]=np.zeros(nx_rss)


# In[21]:


#
# Which are the fibers for which the scale is odd
#
#mask_fe = phot_PS['aperture_sum']/mean_scale > 5.65*f_e
#mask_fe = (slice_r>0.5*f_e) 
mask_scale = (np.log10(fib_scale)<np.log10(med_scale*s)) | (np.log10(fib_scale)>np.log10(med_scale/s)) 
#& (np.log10(fib_scale)<np.log10(med_scale*s))
mask_fe = (flux_ref>15*f_e) 
mask_final = mask_fe & mask_scale


fig = plt.figure(figsize=(5,5))
#ax0 = fig.add_subplot(121)
cm = plt.cm.get_cmap('Spectral')
#im0=ax0.scatter(pt_x,pt_y,c=np.log10(fib_scale),\
#              cmap=cm, vmin=np.log10(med_scale*s), vmax=np.log10(med_scale/s),\
#             s=140,alpha=0.7)
#ax0.scatter(pt_x[mask_scale],pt_y[mask_scale],color='None',edgecolor='black')
#ax0.set_aspect('equal', adjustable='box')
#ax0.set_xlim(-45,45)
#ax0.set_ylim(-45,45)
ax1 = fig.add_subplot(111)
cm = plt.cm.get_cmap('Spectral')
im1=ax1.scatter(pt_x,pt_y,c=np.log10(flux_ref),              cmap=cm, vmin=np.log10(med_flux_ref*s), vmax=np.log10(med_flux_ref/s),             s=140,alpha=0.7)
#ax1.scatter(pt_x[mask_fe],pt_y[mask_fe],color='None',edgecolor='black')
try:
    ax1.scatter(pt_x[mask_final],pt_y[mask_final],color='None',edgecolor='black')
except:
    print('no masked fib')
try:
    ax1.scatter(pt_x[mask_broken],pt_y[mask_broken],s=50,marker='+',color='blue',edgecolor='blue')
except:
    print('no broken fib')
#print(pt_x[mask_broken])

ax1.set_aspect('equal', adjustable='box')
ax1.set_xlim(-45,45)
ax1.set_ylim(-45,45)

#div0 = make_axes_locatable(ax0)
#cax0 = div0.append_axes('right', size='5%', pad=0.05)
#fig.colorbar(im0, cax=cax0, orientation='vertical')

div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax1, orientation='vertical')
plt.savefig(f'{fdir_out}/{name}.broken_fib.png')






# In[22]:


rss_scale=np.ones((ny_rss,nx_rss))
rss_scale[0:331,:]=rel_scale_obj1
rss_scale[331:662,:]=rel_scale_obj2
rss_scale[662:993,:]=rel_scale_obj3

rss_data=rss_data/rss_scale


# In[23]:


#
# We look for those spectra affected by vignetting and broken fibers
#
zero_slice = np.zeros((len(pt_x)))
ind_spec = np.arange(0,len(pt_x))
#print(zero_slice.shape)
for i in arange(0,len(pt_x)):
    #zero_slice[i]=
    zero_slice[i]=rss_data[i,:].tolist().count(0)

mask_vig = (zero_slice>10)
#print(len(zero_slice[mask_vig]))
#print(points[:,0])
nk=6
dist_vig, ind_vig = do_kdtree(points[mask_vig],points[~mask_vig],k=nk)
#print(ind_vig)
print(len(points[mask_vig]))


# In[24]:


#
# We look for those spectra affected by vignetting and broken fibers
#
n_ind_vig=ny_rss
iter_now=0
while (n_ind_vig>0):

    zero_slice = np.zeros((len(pt_x)))
    for i in arange(0,len(pt_x)):
        zero_slice[i]=rss_data[i,:].tolist().count(0)
    #
    # Broken fibers
    #
#    zero_slice[mask_broken]=1000
    mask_vig = (zero_slice>10)
    nk=4
    dist_vig, ind_vig = do_kdtree(points[~mask_vig],points[mask_vig],k=nk)
    ind_spec_vig = ind_spec[mask_vig]
    ind_spec_nvig = ind_spec[~mask_vig]
#    print(len(ind_spec_vig))
    for i_vig in arange(len(ind_spec_vig)):
        n_spec_vig=ind_spec_vig[i_vig]
        (ny_rss,nx_rss)=rss_data.shape
        spec_now = rss_data[n_spec_vig,:]
        med=np.nanmedian(spec_now[spec_now!=0])
        #max_v=np.nanmax(spec_now[spec_now!=0])
        rss_near=np.zeros([nk,nx_rss],dtype=np.float32)
        weights=[]
        for j in arange(0,nk):
            n_spec_nvig=ind_vig[i_vig,j]
            med_j=np.nanmedian(rss_data[n_spec_nvig,:][spec_now!=0])  
            if n_spec_vig in mask_broken:
                spec_scaled = rss_data[n_spec_nvig,:]
                chi_sq=np.ones(nx_rss)
            else:
                spec_scaled = rss_data[n_spec_nvig,:]*(med/med_j)
                chi_sq = np.nansum(((spec_now-spec_scaled)**2/spec_now)[spec_now!=0])
            weights.append(1/chi_sq)
            rss_near[j,:]=spec_scaled
        weights=np.array(weights)
        spec_near=np.average(rss_near,axis=0,weights=weights)
        spec_now[spec_now==0]=spec_near[spec_now==0]
        rss_data[n_spec_vig,:]=spec_now
#        plt.plot(wavelength,spec_now)
#        plt.show()
    rss_data=nan_to_num(rss_data, nan=0.0)  
    n_ind_vig=len(ind_vig)
    iter_now=iter_now+1
    print(f'cleaning: {iter_now} {n_ind_vig}')


# In[25]:


#print(len(ind_vig))


# In[ ]:





# In[26]:


#print(len(ind_spec_vig))
#print(len(ind_spec_nvig))
#print(len(ind_vig))


# i_vig=0
# n_spec_vig=ind_spec_vig[i_vig]
# 
# fig = plt.figure(figsize=(14,6))
# ax = fig.add_subplot(111)
# (ny_rss,nx_rss)=rss_data.shape
# #med=np.nanmedian(rss_data[n_spec_vig,int(nx_rss*0.3):int(nx_rss*0.7)])
# #max_v=np.nanmax(rss_data[n_spec_vig,int(nx_rss*0.4):int(nx_rss*0.6)])
# spec_now = rss_data[n_spec_vig,:]
# med=np.nanmedian(spec_now[spec_now!=0])
# max_v=np.nanmax(spec_now[spec_now!=0])
# rss_near=np.zeros([nk,nx_rss],dtype=np.float32)
# weights=[]
# for j in arange(0,nk):
#     n_spec_nvig=ind_vig[i_vig,j]
#     med_j=np.nanmedian(rss_data[n_spec_nvig,:][spec_now!=0])
#     spec_scaled = rss_data[n_spec_nvig,:]*(med/med_j)
#     chi_sq = np.nansum(((spec_now-spec_scaled)**2/spec_now)[spec_now!=0])
#     weights.append(1/chi_sq)
#     rss_near[j,:]=spec_scaled
#     ax.plot(wavelength,rss_near[j,:],alpha=0.3,color='orange')
# weights=np.array(weights)
# spec_near=np.average(rss_near,axis=0,weights=weights)
# ax.plot(wavelength,rss_data[n_spec_vig,:],color='grey',linewidth=5)
# spec_now[spec_now==0]=spec_near[spec_now==0]
# ax.plot(wavelength,spec_near,color='red', linewidth=2)    
# ax.plot(wavelength,spec_now,color='black', linewidth=1)    
# 
#     
# ax.set_xlim(3700,7500)
# print(med,max_v)
# ax.set_ylim(-0.5*med,1.2*max_v)
# #ax.set_ylim(-45,45)
# #print(ind_vig)

# In[27]:



#print(len(ind_vig))
#print(ind_vig)


# #
# # Clean Vignetting
# #
# #i_vig=0
# #n_spec_vig=ind_spec_vig[i_vig]
# for i_vig in arange(len(ind_spec_vig)):
#     n_spec_vig=ind_spec_vig[i_vig]
#     (ny_rss,nx_rss)=rss_data.shape
#     spec_now = rss_data[n_spec_vig,:]
#     med=np.nanmedian(spec_now[spec_now!=0])
#     max_v=np.nanmax(spec_now[spec_now!=0])
#     rss_near=np.zeros([nk,nx_rss],dtype=np.float32)
#     weights=[]
#     for j in arange(0,nk):
#         n_spec_nvig=ind_vig[i_vig,j]
#         med_j=np.nanmedian(rss_data[n_spec_nvig,:][spec_now!=0])
#         spec_scaled = rss_data[n_spec_nvig,:]*(med/med_j)
#         chi_sq = np.nansum(((spec_now-spec_scaled)**2/spec_now)[spec_now!=0])
#         weights.append(1/chi_sq)
#         rss_near[j,:]=spec_scaled
#     weights=np.array(weights)
#     spec_near=np.average(rss_near,axis=0,weights=weights)
#     spec_now[spec_now==0]=spec_near[spec_now==0]
#     rss_data[n_spec_vig,:]=spec_now
#     

# In[28]:


plt.imshow(rss_data)
#rss_data=nan_to_num(rss_data, nan=0.0)
primhdu = fits.PrimaryHDU(data=rss_data,header=hdu_rss[0].header)
hdulist=fits.HDUList([primhdu])
outfile=f'out/{name}.rss_data.fits.gz'
hdulist.writeto(outfile,overwrite=True)


# In[29]:


#fig = plt.figure(figsize=(7,7))
#ax = fig.add_subplot(111)
#cm = plt.cm.get_cmap('Greys')
#ax.scatter(pt_x,pt_y,c=zero_slice,              cmap=cm, vmin=0, vmax=10,             s=140)
#ax.set_aspect('equal', adjustable='box')
#ax.set_xlim(-45,45)
#ax.set_ylim(-45,45)



# In[30]:


#img_org_075,weights = inter_map_IDW(phot_table,radius,[0,0],[1,1],[nx,ny],slope=slope,\
#                            radius_limit=3.5/pix_size)
print(np.nanmean(rss_data))
print(radius)
print(nx,ny,nx_rss)

print('Creating data cube\n');
cube_org,weights_org=inter_cube_IDW(pt_x_spax,pt_y_spax,rss_data,radius,                                    [0,0],[1,1],[nx,ny,nx_rss],                                    mode='inverseDistance', sigma=sigma, radius_limit=r_lim, 
                                    resolution=1.0, min_fibers=3, slope=slope, bad_threshold=0.1)


# In[31]:

print('Creating variance cube\n');
cube_var,weights_var=inter_cube_IDW(pt_x_spax,pt_y_spax,rss_err**2,radius,                                    [0,0],[1,1],[nx,ny,nx_rss],                                    mode='inverseDistance', sigma=sigma, radius_limit=r_lim,
                                    resolution=1.0, min_fibers=3, slope=slope_var, bad_threshold=0.1)


# In[64]:

print('Creating bad pixels cube\n');
cube_bad,weights_bad=inter_cube_IDW(pt_x_spax,pt_y_spax,rss_bad,radius,                                    [0,0],[1,1],[nx,ny,nx_rss],                                    mode='inverseDistance', sigma=sigma, radius_limit=r_lim,
                                    resolution=1.0, min_fibers=3, slope=slope, bad_threshold=0.1)


# In[32]:


#
# Astrometry and flux recalibration!
#


# In[33]:


#
# We get the r-band filter image
#
# 'r': 6215
#wavelength
#idx_r = np.abs(wavelength - 6215).argmin()
#print(idx_r)
#i0=idx_r-100
#i1=idx_r+100
#img_r = np.nanmean(cube_org[i0:i1,:,:],axis=0)

hdr=hdu_rss[0].header.copy()
hdr['CRVAL3']=hdu_rss[0].header['CRVAL1']
hdr['CRPIX3']=hdu_rss[0].header['CRPIX1']
hdr['CDELT3']=hdu_rss[0].header['CDELT1']
#hdr['CRVAL1']=crval1_cal
#hdr['CRPIX1']=crpix1_cal
#hdr['CDELT1']=pix_size
#hdr['CRVAL2']=crpix2_cal
#hdr['CRPIX2']=crval2_cal
#hdr['CDELT2']=pix_size
hdr['CD1_1']=(-1)*pix_size/3600
hdr['CD1_2']=0.0
hdr['CD1_3']=0.0
hdr['CD2_1']=0.0
hdr['CD2_2']=pix_size/3600
hdr['CD2_3']=0.0
hdr['CD3_1']=0.0
hdr['CD3_2']=0
hdr['CD3_3']=hdu_rss[0].header['CDELT1']
hdr['CTYPE1']='RA---TAN' 
hdr['CTYPE2']='DEC--TAN' 
hdr['CTYPE3']='WAVELENGTH' 
hdr['CUNIT1']='deg'
hdr['CUNIT2']='deg' 
hdr['CUNIT3']='Angstrom'
cube_cl = Cube(data=cube_org,wave=wavelength,header=hdr)
img_u = passband_u.getFluxCube(cube_cl)[0]
img_g = passband_g.getFluxCube(cube_cl)[0]
img_r = passband_r.getFluxCube(cube_cl)[0]
#print(img_r)
#print(img_r.shape)
#print(img_r.shape,i0,i1)
#print(np.nanmean(cube_org))


# In[34]:


plt.imshow(img_r)
print(np.mean(img_r))


# In[35]:


get_astro='../analysis/tables/get_astrometry_cross.warp.csv'
tab_astro=ascii.read(get_astro,fill_values=[('BAD', np.nan)],format='csv')
df_s=tab_astro[tab_astro['name']==name]
if (len(df_s)==0):
    df_s=Table()
    df_s.add_column([name],name='name')
    df_s.add_column([240],name='XC1')
    df_s.add_column([240],name='YC1')
    df_s.add_column([37],name='XC')
    df_s.add_column([34],name='XY')
    df_s.add_column([hdr_cube['CRVAL1']],name='CRVAL1_CR')
    df_s.add_column([hdr_cube['CRVAL2']],name='CRVAL2_CR')
ext='png'
band='r'
i_range=5
filename_PS_r = f'{name}_PS_warp_{band}.fits'
fname=f'{fdir_out}/astro_{name}_{band}_img_r.{ext}'
(crval1_cal,crval2_cal, crpix1_cal,crpix2_cal,e_crval1_cal, e_crval2_cal,rat_mean,rat_std,mag_PS, mag_PS_to_CAL,mag_CAL,FWHM_now,chi_now,rat_map)=get_astrometry(img_r,filename_PS_r,name=name,ext=ext, xc_PS=df_s['XC1'][0],yc_PS=df_s['YC1'][0],RA_PS=df_s['CRVAL1_CR'][0] , DEC_PS=df_s['CRVAL2_CR'][0], xc_cal=df_s['XC'][0]/pix_size-radius , yc_cal=df_s['YC'][0]/pix_size-radius,band=band,i_range=i_range,label='org. recons.',FWHM_in=0.7,A_R=Ar_name,fname=fname)


# In[65]:


hdr=hdu_rss[0].header.copy()
hdr['CRVAL3']=hdu_rss[0].header['CRVAL1']
hdr['CRPIX3']=hdu_rss[0].header['CRPIX1']
hdr['CDELT3']=hdu_rss[0].header['CDELT1']
hdr['CRVAL1']=crval1_cal
hdr['CRPIX1']=crpix1_cal
hdr['CDELT1']=(-1)*pix_size/3600
hdr['CRVAL2']=crval2_cal
hdr['CRPIX2']=crpix2_cal
hdr['CDELT2']=pix_size/3600
hdr['CD1_1']=(-1)*pix_size/3600
hdr['CD1_2']=0.0
hdr['CD1_3']=0.0
hdr['CD2_1']=0.0
hdr['CD2_2']=pix_size/3600
hdr['CD2_3']=0.0
hdr['CD3_1']=0.0
hdr['CD3_2']=0
hdr['CD3_3']=hdu_rss[0].header['CDELT1']
hdr['CTYPE1']='RA---TAN' 
hdr['CTYPE2']='DEC--TAN' 
hdr['CTYPE3']='WAVELENGTH' 
hdr['CUNIT1']='deg'
hdr['CUNIT2']='deg' 
hdr['CUNIT3']='Angstrom'

primhdu = fits.PrimaryHDU(data=cube_org,header=hdr)
hdulist=fits.HDUList([primhdu])
file_cube_org=f'out/{name}.cube.fits.gz'
hdulist.writeto(file_cube_org,overwrite=True)





# In[68]:


cube_bad_int=cube_bad/weights_bad
cube_bad_int[cube_bad_int>0]=1
primhdu = fits.PrimaryHDU(data=cube_bad_int.astype(np.uint8),header=hdr)
hdulist=fits.HDUList([primhdu])
file_cube_bad=f'out/{name}.bad.cube.fits.gz'
hdulist.writeto(file_cube_bad,overwrite=True)


# In[37]:


#
# DAR estimation
#

out_prefix=f'{fdir_out}/{name}'
out_DAR_file=f'{out_prefix}.DAR.png'
(spec2_x,spec2_y)=measureDARPeak_py3d(file_cube_org, out_prefix, coadd=20, smooth_poly=-2, start_wave='3900', end_wave='7100',out_vals='1',figure_out=out_DAR_file)


# In[38]:


#
# DAR correction
#
out_scube=DARcorr(cube_org,spec2_x,spec2_y,verbose=1)
primhdu = fits.PrimaryHDU(data=out_scube,header=hdr)
hdulist=fits.HDUList([primhdu])
file_cube_sorg=f'out/{name}.scube.fits.gz'
hdulist.writeto(file_cube_sorg,overwrite=True)


# In[39]:


#
# Assumed input FWHM=1.0"
#
FWHM_in = 1.5
sigma_FWHM=FWHM_in/pix_size/2.354
new_radius=1
aper = CircularAperture(points_WCS, radius)
new_aper = CircularAperture(points_WCS, new_radius)
#aper_center = CircularAperture(points_WCS, new_radius)
img_Gauss=Gaussian2D_map(nx,ny,1,nx*0.5,ny*0.5,sigma_FWHM,sigma_FWHM,0)
phot_Gauss_org = aperture_photometry(img_Gauss, aper,                                           method='subpixel',subpixels=5)
img_Gauss_r,weights_Gauss_r = inter_map_IDW(phot_Gauss_org,radius,[0,0],[1,1],[nx,ny],sigma=sigma,                                            slope=slope,                            radius_limit=r_lim)
img_psf_r = img_Gauss_r[int(0.4*ny):int(0.6*ny),int(0.4*nx):int(0.6*nx)]
plt.imshow(np.log10(img_psf_r), cmap='gray_r', vmin=-10, vmax=-1, origin='lower')


# In[40]:


plt.imshow(np.log10(img_psf_r), cmap='Spectral', vmin=-10, vmax=-1, origin='lower')


# In[41]:


ext='png'
band='r'
i_range=5
filename_PS_r = f'{name}_PS_warp_{band}.fits'
fname=f'{fdir_out}/astro_{name}_{band}_img_dec_r.{ext}'
img_dec_r = deconv_img(img_r,aper,FWHM_input=0.5,slope=slope,smooth=1,clip=True,filter_epsilon=0.00005)
(crval1_cal,crval2_cal, crpix1_cal,crpix2_cal,e_crval1_cal, e_crval2_cal,rat_mean,rat_std,mag_PS, mag_PS_to_CAL,mag_CAL,FWHM_now,chi_now,rat_map)=get_astrometry(img_dec_r,filename_PS_r,name=name,ext=ext, xc_PS=df_s['XC1'][0],yc_PS=df_s['YC1'][0],RA_PS=df_s['CRVAL1_CR'][0] , DEC_PS=df_s['CRVAL2_CR'][0], xc_cal=df_s['XC'][0]/pix_size-radius , yc_cal=df_s['YC'][0]/pix_size-radius,band=band,i_range=i_range,label='org. recons.',FWHM_in=0.7,A_R=Ar_name,fname=fname)


# In[42]:


n_mc=10
plot=0

cube_dec=cube_org.copy()
cube_err=cube_org.copy()
(nz_c,ny_c,nx_c)=cube_dec.shape
nz_0=0
nz_1=nz_c
#
#f_e = 0.005

#nz_0=1132
#nz_1=1135
#plot=1

for iz in arange(nz_0,nz_1):
    cube_mc = np.zeros([n_mc,ny_c,nx_c],dtype=np.float32)
    for i_mc in arange(0,n_mc):
        img_org=cube_org[iz,:,:]+gaussian_filter(np.sqrt(cube_var[iz,:,:]),sigma=fsize/pix_size)*(np.random.normal(size=[ny_c,nx_c]))
        #        img_org=cube_org[iz,:,:]+gaussian_filter(np.sqrt(cube_var[iz,:,:]),sigma=fsize/pix_size)*(np.random.random([ny_c,nx_c])-0.5)

        if (iz==0):
            img_psf_r=deconv_get_psf(img_org,aper,FWHM_input=1)
        img_loop_dec=deconv_img_psf(img_org,img_psf_r,FWHM_input=1,smooth=0,clip=True,filter_epsilon=f_e)       
        flux_ratio = np.sum(img_org)/np.sum(img_loop_dec)
        img_loop_dec = img_loop_dec*flux_ratio
        img_loop_dec[img_org==0]=0
        cube_mc[i_mc,:,:]=img_loop_dec+gaussian_filter(np.sqrt(cube_var[iz,:,:]),sigma=fsize/pix_size)*(np.random.normal(size=[ny_c,nx_c]))/2
#        cube_mc[i_mc,:,:]=img_loop_dec

     
    cube_dec[iz,:,:]=np.nanmean(cube_mc,axis=0)
    cube_err[iz,:,:]=np.nanstd(cube_mc,axis=0)
    print(f'{iz}',end='\r')
    if (plot==1):
        fig,ax = plt.subplots(1,2,figsize=(10,8))
        ax[0].imshow(np.log10(img_org), cmap='gray_r', vmin=-3, vmax=-0.5, origin='lower')
        ax[0].set_xlim([0*nx,1.0*nx])
        ax[0].set_ylim([0*ny,1.0*ny])
        ax[1].imshow(np.log10(img_loop_dec), cmap='gray_r', vmin=-3, vmax=-0.5, origin='lower')
        ax[1].set_xlim([0*nx,1.0*nx])
        ax[1].set_ylim([0*ny,1.0*ny])
        plt.show()


# In[43]:


#
# DAR deconvolved evaluation
#
out_prefix=f'{fdir_out}/{name}.dec.'
out_DAR_file=f'{out_prefix}.dec.DAR.png'
cube_py3d=Cube(data=cube_dec,wave=wavelength,header=hdr)
(spec2_x,spec2_y)=measureDARPeak_data(cube_py3d, out_prefix, coadd=20, smooth_poly=-2, start_wave='3900', end_wave='7100',out_vals='1',figure_out=out_DAR_file)


# In[44]:


cube_dec_org = cube_dec.copy()
cube_dec=DARcorr(cube_dec_org,spec2_x,spec2_y,verbose=1)
cube_dec_org=[]


# In[69]:


cube_err_org = cube_err.copy()
cube_err=DARcorr(cube_err_org,spec2_x,spec2_y,verbose=1)
cube_err_org=[]


# In[71]:


cube_bad_org = cube_bad.copy()
cube_bad_org=cube_org/weights_bad
cube_bad=DARcorr(cube_bad_org,spec2_x,spec2_y,verbose=1)
cube_bad[cube_bad>0]=1
cube_bad=cube_bad.astype(np.uint8)
cube_bad_org=[]
                          
                          


# In[73]:


cube_weights_org=np.ones(cube_dec.shape)*weights_org
cube_weights=DARcorr(cube_weights_org,spec2_x,spec2_y,verbose=1)
cube_weights_org=[]


# In[45]:


#
# DAR deconvolved evaluation test
#
out_prefix=f'{fdir_out}/{name}.dec_test.'
out_DAR_file=f'{out_prefix}.dec_test.DAR.png'
cube_py3d=Cube(data=cube_dec,wave=wavelength,header=hdr)
(spec2_x,spec2_y)=measureDARPeak_data(cube_py3d, out_prefix, coadd=20, smooth_poly=-2, start_wave='3900', end_wave='7100',out_vals='1',figure_out=out_DAR_file)


# In[46]:


cube_cl_dec = Cube(data=cube_dec,wave=wavelength,header=hdr)
img_u_dec = passband_u.getFluxCube(cube_cl_dec)[0]
img_g_dec = passband_g.getFluxCube(cube_cl_dec)[0]
img_r_dec = passband_r.getFluxCube(cube_cl_dec)[0]

cube_cl_err = Cube(data=cube_err,wave=wavelength,header=hdr)
img_u_err = passband_u.getFluxCube(cube_cl_err)[0]
img_g_err = passband_g.getFluxCube(cube_cl_err)[0]
img_r_err = passband_r.getFluxCube(cube_cl_err)[0]


# In[47]:


ext='png'
band='r'
i_range=5
filename_PS_r = f'{name}_PS_warp_{band}.fits'
fname=f'{fdir_out}/astro_{name}_{band}.{ext}'
(crval1_cal_r,crval2_cal_r, crpix1_cal_r,crpix2_cal_r,e_crval1_cal_r, e_crval2_cal_r,rat_mean_r,rat_std_r,mag_PS_r, mag_PS_to_CAL_r,mag_CAL_r,FWHM_now_r,chi_now_r,rat_map_r)=get_astrometry(img_r_dec,filename_PS_r,name=name,ext=ext, xc_PS=df_s['XC1'][0],yc_PS=df_s['YC1'][0],RA_PS=df_s['CRVAL1_CR'][0] , DEC_PS=df_s['CRVAL2_CR'][0], xc_cal=df_s['XC'][0]/pix_size-radius , yc_cal=df_s['YC'][0]/pix_size-radius,band=band,i_range=i_range,label='org. recons.',FWHM_in=0.7,A_R=Ar_name,fname=fname)


# In[48]:


ext='png'
band='g'
i_range=5
filename_PS_g = f'{name}_PS_warp_{band}.fits'
fname=f'{fdir_out}/astro_{name}_{band}.{ext}'
(crval1_cal_g,crval2_cal_g, crpix1_cal_g,crpix2_cal_g,e_crval1_cal_g, e_crval2_cal_g,rat_mean_g,rat_std_g,mag_PS_g, mag_PS_to_CAL_g,mag_CAL_g, FWHM_now_g,chi_now_g,rat_map_g)=get_astrometry(img_g_dec,filename_PS_g,name=name,ext=ext, xc_PS=df_s['XC1'][0],yc_PS=df_s['YC1'][0],RA_PS=df_s['CRVAL1_CR'][0] , DEC_PS=df_s['CRVAL2_CR'][0], xc_cal=df_s['XC'][0]/pix_size-radius, yc_cal=df_s['YC'][0]/pix_size-radius,band=band,i_range=i_range,label='org. recons.',FWHM_in=0.7,A_R=Ag_name,fname=fname)


# In[81]:


# Using the g-band ratio produces images of poorer quality!
#rat_map = 0.5 * (rat_map_g+rat_map_r)
rat_map=rat_map_r
flux_scale = 0.5*(rat_mean_g+rat_mean_r)
print(f'MAG: g={mag_PS_g}/{mag_CAL_g}, r={mag_PS_r}/{mag_CAL_r}')
print(f'RAT: g=={rat_mean_g}+-{rat_std_g} ; r=={rat_mean_r}+-{rat_std_r}')
print(f'FWHM: g=={FWHM_now_g}; r=={FWHM_now_r}')
hdr=hdu_rss[0].header.copy()
hdr['CRVAL3']=hdu_rss[0].header['CRVAL1']
hdr['CRPIX3']=hdu_rss[0].header['CRPIX1']
hdr['CDELT3']=hdu_rss[0].header['CDELT1']
hdr['CRVAL1']=crval1_cal_r
hdr['CRPIX1']=crpix1_cal_r
hdr['CDELT1']=(-1)*pix_size/3600
hdr['CRVAL2']=crval2_cal_r
hdr['CRPIX2']=crpix2_cal_r
hdr['CDELT2']=pix_size/3600
hdr['CD1_1']=(-1)*pix_size/3600
hdr['CD1_2']=0.0
hdr['CD1_3']=0.0
hdr['CD2_1']=0.0
hdr['CD2_2']=pix_size/3600
hdr['CD2_3']=0.0
hdr['CD3_1']=0.0
hdr['CD3_2']=0
hdr['CD3_3']=hdu_rss[0].header['CDELT1']
hdr['CTYPE1']='RA---TAN' 
hdr['CTYPE2']='DEC--TAN' 
hdr['CTYPE3']='WAVELENGTH' 
hdr['CUNIT1']='deg'
hdr['CUNIT2']='deg' 
hdr['CUNIT3']='Angstrom'
hdr['HIERARCH PIPE P1 PIPE REGISTER']=1
hdr['HIERARCH PIPE P2 PIPE REGISTER']=1 
hdr['HIERARCH PIPE P3 PIPE REGISTER']=1 
hdr['HIERARCH PIPE P1 PIPE SCALE']=rel_scale_obj1
hdr['HIERARCH PIPE P2 PIPE SCALE']=rel_scale_obj2
hdr['HIERARCH PIPE P3 PIPE SCALE']=rel_scale_obj3
hdr['HIERARCH PIPE REGISTER']=int(already_reg)
hdr['HIERARCH PIPE REGISTER_SDSS']=int(already_reg)
hdr['HIERARCH PIPE REGISTER_PS']=int(np.abs(1-already_reg))
hdr['HIERARCH PIPE RAT_PS_g']=rat_mean_g
hdr['HIERARCH PIPE RAT_STD_g']=rat_std_g
hdr['HIERARCH PIPE RAT_PS_r']=rat_mean_r
hdr['HIERARCH PIPE RAT_STD_r']=rat_std_r
hdr['HIERARCH PIPE RAT_PS']=flux_scale
hdr['HIERARCH PIPE FLUX_SCALED']=1
hdr['HIERARCH PIPE FWHM_g']=FWHM_now_g
hdr['HIERARCH PIPE FWHM_r']=FWHM_now_g
hdr['HIERARCH PIPE mag_PS_g']=mag_PS_g
hdr['HIERARCH PIPE mag_CAL_g']=mag_CAL_g
hdr['HIERARCH PIPE mag_PS_r']=mag_PS_r
hdr['HIERARCH PIPE mag_CAL_r']=mag_CAL_r
hdr['HIERARCH PIPE VERS']=2.3


# In[50]:


xc_h=nx/2
yc_h=ny/2
r_h=80/2/pix_size
a=0.5*r_h
b=(np.sqrt(3)/2)*r_h
poly_verts = [(-r_h+xc_h,yc_h), (-a+xc_h,b+yc_h), (a+xc_h,b+yc_h),(r_h+xc_h,0+yc_h),(a+xc_h,-b+yc_h),(-a+xc_h,-b+yc_h)]
x, y = np.meshgrid(np.arange(nx), np.arange(ny))
x, y = x.flatten(), y.flatten()
points_h = np.vstack((x,y)).T
path_h = mpl_path(poly_verts)
grid_h = path_h.contains_points(points_h)
grid_h = grid_h.reshape((ny,nx))


# In[51]:


#print(rat_map.shape)

#
# S/N map
#
SN_map_g = img_g_dec/img_g_err

#points_inside_poly

mask_good_pix = (SN_map_g<1.0) | (img_g_dec< 0.1*f_e) #| (rat_map>1.5) | (rat_map<0.1)
#mask_good_pix = (SN_map_g<1.5) | (img_g_dec< 0.2*f_e) #| (rat_map>1.5) | (rat_map<0.1)
#mask_good_pix = (SN_map_g<0.5) | (img_g_dec< 0.5*f_e) | (rat_map>4) | (rat_map<1/4)
#mask_good_pix = grid_h # | (SN_map_g<0.5) | (img_g_dec< 0.1*f_e)
rat_map_inv = 1/rat_map
rat_map_inv[mask_good_pix]=1
rat_map_inv[~grid_h]=0

if (plot==1):
    fig,ax=subplots(2,2,figsize=(8,8))
    im0=ax[0][0].imshow(rat_map, vmin=0,vmax=2,cmap='Greys', origin='lower')
    im1=ax[0][1].imshow(SN_map_g, cmap='Greys',norm=PowerNorm(0.25), origin='lower')
    im2=ax[1][0].imshow(img_g_dec, cmap='Greys',norm=PowerNorm(0.25), origin='lower')
    im2=ax[1][1].imshow(grid_h, cmap='Greys', origin='lower')

    div0 = make_axes_locatable(ax[0][0])
    cax0 = div0.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax0, orientation='vertical')

    div1 = make_axes_locatable(ax[0][1])
    cax1 = div1.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax1, orientation='vertical')

    div2 = make_axes_locatable(ax[1][0])
    cax2 = div2.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax2, orientation='vertical')


# In[82]:





cube_dec_flat=cube_dec*rat_map_inv/flux_scale
primhdu = fits.PrimaryHDU(data=cube_dec_flat,header=hdr)

#
# We create the mask of the stars
#
CAT_GAIA_NEAR = f'{fdir_gaia}/gaia_dr3_sn5.{name}.pkl'
gaia_mask_org=get_mask_gaia(primhdu,name,CATALOG_PKL=CAT_GAIA_NEAR,psf=2.5,mag_lim=None,plot=0)  




hdulist=fits.HDUList([primhdu])
hdulist.append(fits.ImageHDU(data=cube_err,name='ERROR'))
hdulist.append(fits.ImageHDU(data=cube_weights,name='ERRWEIGHT'))
hdulist.append(fits.ImageHDU(data=cube_bad,name='BADPIX'))
hdulist.append(fits.ImageHDU(data=rat_map_inv/flux_scale,name='FLAT'))
hdulist.append(fits.ImageHDU(data=gaia_mask_org.data,name='GAIA_MASK'))
outfile=f'out/{name}.V500.drscube.fits.gz'
hdulist.writeto(outfile,overwrite=True)
print(f'file {outfile} created')


end_time = time.time()

total_time = end_time - start_time
print('\n Total Time: '+str(total_time))
