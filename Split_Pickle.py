#!/usr/bin/env python3
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

def make_region_string(stars,catalog, psf):
    top_string = """         
    # Region file format: DS9 version 4.1
    global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
    fk5
    """
    for k in stars:
        ra  = catalog[k,1]
        dec = catalog[k,2]
        mag = catalog[k,3]
        #top_string+=f"circle({ra},{dec},0.001)  # text={{mag:{mag}}}\n"
        top_string+=f'circle({ra},{dec},{psf}")\n'
    return top_string

def find_centre(image,xg,yg,psf_im, plot, gname, pix_scale=2,delta=1.5):
    xgr = int(round(xg))
    ygr = int(round(yg))
    sig = round(psf_im/2.354)
    print(f'find_centre_in = {xgr},{ygr},{sig}')
    xs1=xgr-int(delta*sig/pix_scale)
    xs2=xgr+int(delta*sig/pix_scale)
    ys1=ygr-int(delta*sig/pix_scale)
    ys2=ygr+int(delta*sig/pix_scale)
    
    if xs1<0:
        xs1=0
    if xs2>image.shape[1]:
        xs2 = image.shape[1]
        
    if ys1<0:
        ys1=0
    if ys2>image.shape[0]:
        ys2 = image.shape[0]
    
    cutout = image[ys1:ys2, xs1:xs2]
    
    
    
    #fit_w = fitting.LevMarLSQFitter()
    #gmod = models.Gaussian2D(cutout[sig,sig], sig, sig, sig, sig)
    cutout=cutout-np.nanmin(cutout)
    yi, xi = numpy.indices(cutout.shape)
    x = np.arange(0, cutout.shape[1])
    y = np.arange(0, cutout.shape[0])
    xx, yy = np.meshgrid(x, y)

    g_x=np.nansum(xx*cutout**2)/np.nansum(cutout**2)
    g_y=np.nansum(yy*cutout**2)/np.nansum(cutout**2)
    x2 = np.nansum(cutout*xx**2)/np.nansum(cutout)-g_x**2
    y2 = np.nansum(cutout*yy**2)/np.nansum(cutout)-g_y**2
    xy = np.nansum(cutout*xx*yy)/np.nansum(cutout)-g_x*g_y
    A=np.sqrt((x2+y2)/2+np.sqrt(((g_x**2-g_y**2)/2)**2+xy**2))
    B=np.sqrt((x2+y2)/2-np.sqrt(((g_x**2-g_y**2)/2)**2+xy**2))

    #print(f'A,B = {A},{B}')
    
    #print(gmod,xi,yi,cutout)
#    g = fit_w(gmod, xi, yi, cutout)#, np.sqrt(np.abs(cutout)))
    #print(w)
    #model_data = g(xi, yi)
    
    xfit = xgr-sig + g_x
    yfit = ygr-sig + g_y
    
    x_sigma = A
    y_sigma = B

    
    if plot >0:
        fig, ax = plt.subplots()
        ax.imshow(cutout,origin='lower')
        ax.plot(g_x+0.5*pix_scale,g_y+0.5*pix_scale,'ro', label = 'Fit')
        ax.plot(xg-(xgr-delta*sig/pix_scale),yg-(ygr-delta*sig/pix_scale),'bo', label = 'Gaia')
        fig.legend()
        plt.tight_layout()
    if plot == 2:
        plt.savefig(f'{gname}.fit_star_{xgr}_{ygr}.png')

    
    return xfit, yfit, x_sigma, y_sigma
    

def get_mask_gaia(hdu,gname,CATALOG_PKL='../Artemi/gaia_dr3_sn5.pkl',psf=2.5,mag_lim=None,plot=1):


    header = hdu.header
    cube = hdu.data
    wave = numpy.linspace(header['CRVAL3'],header['CRVAL3']+header['NAXIS3']*header['CD3_3'],header['NAXIS3'])
    wslice = (wave > 6000) * (wave < 6500)

    image = numpy.sum(cube[wslice,:,:],0)
    w = WCS(header)
    w = w.dropaxis(2)

    pix_scale=abs(header['CDELT1']*3600)
    print(f'pix_scale = {pix_scale}')

    gaia_num = pickle.load( open(CATALOG_PKL, "rb") )


    centre_im = numpy.array(image.shape)/2
    centre_w  = w.pixel_to_world(centre_im[0],centre_im[1])
    orig_w    = w.pixel_to_world(0,0)

    dd = w.pixel_to_world(0,0)
    uu = w.pixel_to_world(image.shape[0],image.shape[1])
    du = w.pixel_to_world(0,image.shape[1])
    ud = w.pixel_to_world(image.shape[0],0)

    ra_max  = max(dd.ra.deg, uu.ra.deg, ud.ra.deg, du.ra.deg)
    dec_max = max(dd.dec.deg, uu.dec.deg, ud.dec.deg, du.dec.deg)

    ra_min  = min(dd.ra.deg, uu.ra.deg, ud.ra.deg, du.ra.deg)
    dec_min = min(dd.dec.deg, uu.dec.deg, ud.dec.deg, du.dec.deg)

    mask_ra = ( ra_min < gaia_num[:,1] ) * ( gaia_num[:,1] < ra_max )
    mask_dec = ( dec_min < gaia_num[:,2] ) * ( gaia_num[:,2] < dec_max )
    if mag_lim != None:
        mask_mag = ( gaia_num[:,3] < mag_lim )

    if mag_lim == None:
        mask_def = mask_ra*mask_dec
    else:
        mask_def = mask_ra*mask_dec*mask_mag

    stars = numpy.where(mask_def)[0]

    if len(stars) > 0:
        reg_str = make_region_string(stars,gaia_num, psf)

        r = pyregion.parse(reg_str)
        r2 = r.as_imagecoord(hdu.header)
        patch_list, artist_list = r2.get_mpl_patches_texts()
        hdu_out = fits.PrimaryHDU(data=image)
        hdu_out.header.extend(w.to_header())
        gmask = r2.get_mask(hdu=hdu_out)

        if plot > 0:
            fig, ax = plt.subplots(1,2,figsize = (8,6))
            ax1=ax[0]
            ax1.imshow(image,cmap='Greys_r', origin='lower')

            for p in patch_list:
                ax1.add_patch(p)
            for t in artist_list:
                ax1.add_artist(t)
            ax[1].imshow(gmask, cmap='Greys_r',origin='lower')
            plt.tight_layout()
            if plot == 2:
                plt.savefig('gaia-det.'+gname+'.png')
    else: 
        gmask=np.zeros((header['NAXIS2'],header['NAXIS1']))


    star_file = open('gaia-stars.'+gname+'.txt','w')
    star_file.write('Index\tRa\tDec\tX\tY\tXfit\tYfit\tXYerr (")\tMag\tRad (")\n')
    psf_im = (psf/3600.)/abs(w.pixel_scale_matrix[0,0])
    fwhm=[]
    for s in stars:
        index = gaia_num[s,0]
        ra = gaia_num[s,1]
        dec = gaia_num[s,2]
        mag = gaia_num[s,3]
        xx,yy = w.world_to_pixel(crd.SkyCoord(ra,dec,unit='deg'))
        xf, yf, x_sig, y_sig = find_centre(image,float(xx),float(yy),psf_im, \
                                           plot, gname,pix_scale=pix_scale, delta=2)
        sig=np.nanmean([x_sig,y_sig])
        fwhm_now = 2.354*pix_scale*sig
        fwhm.append(fwhm_now)
        dxy = crd.SkyCoord(ra,dec,unit='deg').separation( w.pixel_to_world(xf,yf) )
        rad = psf
        star_file.write(f'{index:.0f}\t{ra:.4f}\t{dec:.4f}\t{xx:.2f}\t{yy:.2f}\t{xf:.2f}\t{yf:.2f}\t{dxy.arcsec:.2e}\t{mag:.2f}\t{rad:.2f}\n')
    star_file.close()
    if plot == 1:
        plt.show()
    #else:
    hdu_out = fits.PrimaryHDU(data=gmask*1)
    hdu_out.header.extend(w.to_header())
    hdu_out.header['N_STARS']=len(stars)
    if (len(stars)>0):
        fwhm=np.array(fwhm)
        fwhm_mean=np.nanmean(fwhm)
        fwhm_std=np.nanstd(fwhm)

        fwhm_mean=np.nan_to_num(fwhm_mean)
        fwhm_std=np.nan_to_num(fwhm_mean)
        hdu_out.header['FWHM_STARS_MEAN']=fwhm_mean
        hdu_out.header['FWHM_STARS_STD']=fwhm_std
    else:
        hdu_out.header['FWHM_STARS_MEAN']=0
        hdu_out.header['FWHM_STARS_STD']=0
    return(hdu_out)
    
#    fits.writeto('gaia-mask.'+gname+'.fits',image*0, header = hdu_out.header, overwrite=True)


get_astro='../analysis/tables/get_astrometry_cross.warp.csv'
tab_astro=ascii.read(get_astro,fill_values=[('BAD', np.nan)],format='csv')
CATALOG_PKL='../Artemi/gaia_dr3_sn5.pkl'
gaia_num = pickle.load( open(CATALOG_PKL, "rb") )

for tab_now in tab_astro:
    print(tab_now['name'],tab_now['CRVAL1_CR'], tab_now['CRVAL2_CR'])
    delta = 5/60
    name = tab_now['name']
    CAT_GAIA_NEAR = f'gaia/gaia_dr3_sn5.{name}.pkl'
    if (path.exists(CAT_GAIA_NEAR)):
        print(f'{CAT_GAIA_NEAR} exists')
    else:
        ra_min = tab_now['CRVAL1_CR']-delta
        ra_max = tab_now['CRVAL1_CR']+delta
        dec_min = tab_now['CRVAL2_CR']-delta
        dec_max = tab_now['CRVAL2_CR']+delta
        mask_ra = ( ra_min < gaia_num[:,1] ) * ( gaia_num[:,1] < ra_max )
        mask_dec = ( dec_min < gaia_num[:,2] ) * ( gaia_num[:,2] < dec_max )
        mask_def = mask_ra*mask_dec
        gaia_num_near = gaia_num[mask_def]
        file = open(CAT_GAIA_NEAR, 'wb')
        pickle.dump(gaia_num_near, file)
        file.close()

