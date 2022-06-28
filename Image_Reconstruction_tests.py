#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

#import naturalneighbor

from astropy.modeling.models import Gaussian2D
from skimage import color, data, restoration
from mpl_toolkits.axes_grid1 import make_axes_locatable

from astropy.modeling import models,fitting

#import _ngl
#from photutils.aperture import aperture_photometry


# In[2]:


def do_kdtree_n(X):
    mytree = KDTree(X)    
    dist, ind = mytree.query(X[:1], k=3)
    print(dist,ind)
    return np.array(dist),np.array(ind)

def do_kdtree(combined_x_y_arrays,points,k=1):
    mytree = cKDTree(combined_x_y_arrays)
    dist, indexes = mytree.query(points,k=k)
    return np.array(dist),np.array(indexes)


# In[3]:


#pt_file="UGC09777.pt"
#pt_file="data/UGC02225.mos.V500.pt.txt"
pt_file="/disk-a/sanchez/ppak/legacy/DATA/SSP/legacy/mos_new.pt.txt"
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


# In[5]:


fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111)
print(pt_shape[1])
for indx,point in enumerate(points):
#    color='blue'
    ax.add_patch(plt.Circle((point[0],point[1]), radius=float(pt_shape[1]), edgecolor='black',color='blue', alpha=0.6))
ax.scatter(pt_x,pt_y,s=10,edgecolor='blue',alpha=0.3)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-45,45)
ax.set_ylim(-45,45)

ax.set_xlim(-5,5)
ax.set_ylim(-5,5)


# In[6]:


filename = 'NGC5947_PS_warp_r.fits'#NGC5947_PS_g.fits'
hdu = fits.open(filename)[0]





wcs = WCS(hdu.header)
#wcs = wcs.dropaxis(2)
fig,ax = plt.subplots(figsize=(10,5))
ax=plt.subplot()
pix_size=np.abs(wcs.wcs.cdelt[1]*3600)
#pix_rat = np.abs(wcs.wcs.cdelt[0]*3600)
EXPTIME = hdu.header['EXPTIME']
ww_now=4866
f_ratio = 2*((10**(-6.64))/EXPTIME)*(((1e-23)*(3e18))/ww_now**2)*1e16/pix_size**2
hdu.data=hdu.data*f_ratio
ax.imshow(np.log10(hdu.data), cmap='gray_r', vmin=-3, vmax=0.15, origin='lower')#imshow(hdu.data, origin='lower')
(ny,nx)=hdu.data.shape
#ax.scatter(232.652436800918,42.7171665852056, transform=ax.get_transform('fk5'), s=50,
#            edgecolor='white', facecolor='none')

points_WCS=np.zeros((len(pt_x),2))

points_WCS[:,0]=nx/2-pt_x/(wcs.wcs.cdelt[0]*3600)
points_WCS[:,1]=ny/2+pt_y/(wcs.wcs.cdelt[1]*3600)
#print(points_WCS)
#print(pt_shape[1])
radius=float(pt_shape[1])/pix_size
#print('radius=',radius)
#print(nx,ny)
for indx,point in enumerate(points_WCS):
#    color='blue'
    ax.add_patch(plt.Circle((point[0],point[1]), radius=radius,                            edgecolor='red',facecolor='none', alpha=0.4))



plt.xlabel('R.A.')
plt.ylabel('Dec')
print("DONE")


# In[7]:


aper = CircularAperture(points_WCS, radius)
phot_table = aperture_photometry(hdu.data, aper, method='subpixel',subpixels=5)
#print(hdu.data.shape)


# In[8]:


print(phot_table)


# In[9]:




# In[11]:


def inter_map_IDW(phot_table_in,radius,crpix,cdelt,dim,                  mode='inverseDistance', sigma=1.0, radius_limit=5,                   resolution=1.0, min_fibers=3, slope=2.0, bad_threshold=0.1):
    phot_table=phot_table_in.copy()
    img = np.zeros(dim,dtype=np.float32)
    weights = np.zeros(img.shape, dtype=np.float32)
    xi = crpix[0]+cdelt[0]*np.arange(0,dim[0])
    yi = crpix[1]+cdelt[1]*np.arange(0,dim[1])
    xi,yi = np.meshgrid(xi,yi)
    phot_table['aperture_sum']=phot_table['aperture_sum']/(np.pi*radius**2)
    for phot_now in phot_table:
        if (phot_now['aperture_sum']>0):
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
    
def inter_map(phot_table_in,radius,crpix,cdelt,dim,method='cubic'):
    phot_table=phot_table_in.copy()
    xi = crpix[0]+cdelt[0]*np.arange(0,dim[0])
    yi = crpix[1]+cdelt[1]*np.arange(0,dim[1])
    xi,yi = np.meshgrid(xi,yi)
    phot_table['aperture_sum']=phot_table['aperture_sum']/(np.pi*radius**2)
    img = griddata((phot_table['xcenter'].value, phot_table['ycenter'].value),                   phot_table['aperture_sum'].value, (xi, yi),method=method)
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


# In[12]:


img,weights = inter_map_IDW(phot_table,radius,[0,0],[1,1],[480,480],slope=0.75,                            radius_limit=3.5/pix_size)
#img_map = inter_map(phot_table,radius,[0,0],[1,1],[480,480],method='cubic')


# In[13]:


img_map = inter_map(phot_table,radius,[0,0],[1,1],[480,480],method='cubic')

#interpolate.interp2d(x, y, z, kind='cubic')


# In[14]:


def plot_three(img_in,img_now,res):
    fig,ax = plt.subplots(1,3,figsize=(10,8))
    plt.subplot(ax[0],projection=wcs)
    ax[0].imshow(np.log10(img_in), cmap='magma_r', vmin=-3, vmax=0.15, origin='lower')
    plt.subplot(ax[1],projection=wcs)
    ax[1].imshow(np.log10(img_now), cmap='magma_r', vmin=-3, vmax=0.15, origin='lower')
    plt.subplot(ax[2],projection=wcs)
    std=np.nanstd(res)
    im=ax[2].imshow(res/img_in, cmap='coolwarm', vmin=-1, vmax=1, origin='lower')


    ax[0].set_xlim([0.2*nx,0.8*nx])
    ax[0].set_ylim([0.2*ny,0.8*ny])

    ax[1].set_xlim([0.2*nx,0.8*nx])
    ax[1].set_ylim([0.2*ny,0.8*ny])

    ax[2].set_xlim([0.2*nx,0.8*nx])
    ax[2].set_ylim([0.2*ny,0.8*ny])
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im,cax=cax)
#    plt.tight_layout()
    plt.show()


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
        #plot_three(s_img,img_test,s_img-img_test)
    return FWHM_final,sigma_min


# #
# # We test the FWHM of the original image...
# #

# In[15]:


FWHMs=[]
#slopes=arange(0.75,4.0,0.25)
slopes=arange(0.75,1.25,0.25)
for slope_now in slopes:
    img,weights = inter_map_IDW(phot_table,radius,[0,0],[1,1],[480,480],slope=slope_now,                                radius_limit=0.5*radius/pix_size)
    print('Slope =',slope_now)
    fwhm_now,sigma_min = get_FWHM_conv(img,hdu.data,FWHM_ref_arc=1,pix_size=pix_size,plot=1)
    FWHMs.append(fwhm_now)
FWHMs=np.array(FWHMs)


# In[16]:


plt.scatter(slopes,FWHMs)


# In[17]:


#
# Now we test the deconvolution
#


# In[15]:


def Gaussian2D_map(nx,ny,A,xc,yc,sigma_x,sigma_y,theta=0):
    xi = np.arange(0,nx)
    yi = np.arange(0,ny)
    xi,yi = np.meshgrid(xi,yi)    
    img = Gaussian2D(A, xc, yc, sigma_x, sigma_y, theta=theta)(xi, yi)
    return img

def deconv_img(img_center,aper_center,FWHM_input=1,slope=0.75,smooth=0):
    sigma_FWHM=FWHM_input/pix_size/2.354
    (ny,nx)=img_center.shape
    img_Gauss=Gaussian2D_map(nx,ny,1,nx*0.5,ny*0.5,sigma_FWHM,sigma_FWHM,0)
    phot_Gauss = aperture_photometry(img_Gauss, aper_center,                                           method='subpixel',subpixels=5)
    img_Gauss_r,weights_Gauss_r = inter_map_IDW(phot_Gauss,radius,[0,0],[1,1],[nx,ny],slope=slope,                                                radius_limit=0.5*5/pix_size)
    img_psf_r = img_Gauss_r[int(0.4*ny):int(0.6*ny),int(0.4*nx):int(0.6*nx)]
    img_loop_in = img_center+img_center.max() * 1E-5 * np.random.random(img_center.shape)
    img_loop_dec = restoration.richardson_lucy(img_loop_in, img_psf_r,10)#, num_iter=30)
    if (smooth==1):
        img_loop_dec = gaussian_filter(img_loop_dec,sigma=FWHM_input/pix_size/2.354)#*np.sum(img_Gauss)
    flux_ratio = np.sum(img_center)/np.sum(img_loop_dec)
    img_loop_dec = img_loop_dec*flux_ratio
    img_loop_dec[img_center==0]=0
    return img_loop_dec


# In[16]:


tri = Delaunay(points)
cx=[]
cy=[]
dist=[]
for j, s in enumerate(tri.simplices):
    p0_x = 0.5*(points[s][0][0]+points[s][1][0])
    p1_x = 0.5*(points[s][2][0]+points[s][1][0])
    p2_x = 0.5*(points[s][0][0]+points[s][2][0])
    p0_y = 0.5*(points[s][0][1]+points[s][1][1])
    p1_y = 0.5*(points[s][2][1]+points[s][1][1])
    p2_y = 0.5*(points[s][0][1]+points[s][2][1])
    cx.append(p0_x)
    cy.append(p0_y)
    cx.append(p1_x)
    cy.append(p1_y)
    cx.append(p2_x)
    cy.append(p2_y)
    p = points[s].mean(axis=0)
    dist_p1 = np.sqrt((p[0]-points[s][0][0])**2+(p[1]-points[s][0][1])**2)
    dist_p2 = np.sqrt((p[0]-points[s][1][0])**2+(p[1]-points[s][1][1])**2)
    dist_p3 = np.sqrt((p[0]-points[s][2][0])**2+(p[1]-points[s][2][1])**2)
    dist_now=np.max([dist_p1,dist_p2,dist_p3])
    dist.append(dist_now)
#    print(cx,cy,dist_p3)
cx=np.array(cx)
cy=np.array(cy)

#mid_points=np.zeros((len(cx),2))

#mid_points=unique2d(mid_points)


mid_points_WCS=np.zeros((len(cx),2))
mid_points_WCS[:,0]=nx/2-cx/(wcs.wcs.cdelt[0]*3600)
mid_points_WCS[:,1]=ny/2+cy/(wcs.wcs.cdelt[1]*3600)

mid_points_WCS=unique2d(mid_points_WCS)

print(len(cx),len(mid_points_WCS))
#
# We clean those points that are too nearby!
#

dist=np.array(dist)
mean_dist=np.around(np.mean(dist),1)
print(mean_dist)



# In[21]:


aper = CircularAperture(points_WCS, radius)
slope_now=1.25
img,weights = inter_map_IDW(phot_table,radius,[0,0],[1,1],[480,480],slope=slope_now,                            radius_limit=0.5*radius/pix_size)
print('Slope =',slope_now)
img_dec = deconv_img(img,aper,FWHM_input=1,slope=slope_now)
fwhm_now,sigma_min = get_FWHM_conv(img_dec,hdu.data,FWHM_ref_arc=1,pix_size=pix_size,plot=1)

points_WCS_shift=points_WCS.copy()
points_WCS_shift[0]=points_WCS[0]+10

aper_shift = CircularAperture(points_WCS_shift, radius)
img,weights = inter_map_IDW(phot_table,radius,[0,0],[1,1],[480,480],slope=slope_now,                            radius_limit=0.5*radius/pix_size)
print('Slope =',slope_now)
img_dec = deconv_img(img,aper_shift,FWHM_input=1,slope=slope_now)
fwhm_now,sigma_min = get_FWHM_conv(img_dec,hdu.data,FWHM_ref_arc=1,pix_size=pix_size,plot=1)

plt.imshow(img_dec,origin='lower')

print("PASO!!!");
quit()

# In[20]:


#
# No smooth
#

aper = CircularAperture(points_WCS, radius)
FWHMs_dec=[]
slopes=arange(0.75,4.0,0.25)
for slope_now in slopes:
    img,weights = inter_map_IDW(phot_table,radius,[0,0],[1,1],[480,480],slope=slope_now,                                radius_limit=0.5*radius/pix_size)
    print('Slope =',slope_now)
    img_dec = deconv_img(img,aper,FWHM_input=1,slope=slope_now)
    fwhm_now,sigma_min = get_FWHM_conv(img_dec,hdu.data,FWHM_ref_arc=1,pix_size=pix_size,plot=1)
    FWHMs_dec.append(fwhm_now)
FWHMs_dec=np.array(FWHMs_dec)


# In[21]:


#
# Smooth
#

aper = CircularAperture(points_WCS, radius)
FWHMs_dec_s=[]
slopes=arange(0.75,4.0,0.25)
for slope_now in slopes:
    img,weights = inter_map_IDW(phot_table,radius,[0,0],[1,1],[480,480],slope=slope_now,                                radius_limit=0.5*radius/pix_size)
    print('Slope =',slope_now)
    img_dec = deconv_img(img,aper,FWHM_input=1,slope=slope_now,smooth=1)
    fwhm_now,sigma_min = get_FWHM_conv(img_dec,hdu.data,FWHM_ref_arc=1,                                       pix_size=pix_size,plot=1)
    FWHMs_dec_s.append(fwhm_now)
FWHMs_dec_s=np.array(FWHMs_dec_s)


# In[22]:


plt.scatter(slopes,FWHMs)
plt.scatter(slopes,FWHMs_dec)
plt.scatter(slopes,FWHMs_dec_s)


# #
# # Further tests!!!
# #

# In[14]:


#nx=480
#ny=480
#img_psf = img_map[int(0.4*ny):int(0.6*ny),int(0.4*nx):int(0.6*nx)]
#img_Gauss_dec = restoration.richardson_lucy(img_map, img_psf, 5)



#img_map_bi = inter_map(phot_table,radius,[0,0],[1,1],[480,480],method='quintic')


# In[15]:


fig,ax = plt.subplots(2,2,figsize=(10,8))
plt.subplot(ax[0][0],projection=wcs)
ax[0][0].imshow(np.log10(hdu.data), cmap='gray_r', vmin=-3, vmax=0.15, origin='lower')
plt.subplot(ax[1][0],projection=wcs)
ax[0][1].imshow(np.log10(hdu.data), cmap='gray_r', vmin=-3, vmax=-3, origin='lower')
cm = plt.cm.get_cmap('Greys')
ax[0][1].scatter(phot_table['xcenter'],phot_table['ycenter'],                 c=np.log10(phot_table['aperture_sum']/(np.pi*radius**2)),              cmap=cm, vmin=-3, vmax=0.15,             s=140)
plt.subplot(ax[1][0],projection=wcs)
ax[1][0].imshow(np.log10(img), cmap='gray_r', vmin=-3, vmax=0.15, origin='lower')

plt.subplot(ax[1][1],projection=wcs)
ax[1][1].imshow(np.log10(img_map), cmap='gray_r', vmin=-3, vmax=0.15, origin='lower')
#ax[1][1].imshow(weights, cmap='gray_r', vmin=0, vmax=np.max(weights), origin='lower')


ax[0][0].set_xlim([0.2*nx,0.8*nx])
ax[0][0].set_ylim([0.2*ny,0.8*ny])

ax[0][1].set_xlim([0.2*nx,0.8*nx])
ax[0][1].set_ylim([0.2*ny,0.8*ny])

ax[1][0].set_xlim([0.2*nx,0.8*nx])
ax[1][0].set_ylim([0.2*ny,0.8*ny])

ax[1][1].set_xlim([0.2*nx,0.8*nx])
ax[1][1].set_ylim([0.2*ny,0.8*ny])


# In[16]:


print(np.max(img),np.max(weights))


# In[17]:


tri = Delaunay(points)
cx=[]
cy=[]
dist=[]
for j, s in enumerate(tri.simplices):
    p0_x = 0.5*(points[s][0][0]+points[s][1][0])
    p1_x = 0.5*(points[s][2][0]+points[s][1][0])
    p2_x = 0.5*(points[s][0][0]+points[s][2][0])
    p0_y = 0.5*(points[s][0][1]+points[s][1][1])
    p1_y = 0.5*(points[s][2][1]+points[s][1][1])
    p2_y = 0.5*(points[s][0][1]+points[s][2][1])
    cx.append(p0_x)
    cy.append(p0_y)
    cx.append(p1_x)
    cy.append(p1_y)
    cx.append(p2_x)
    cy.append(p2_y)
    p = points[s].mean(axis=0)
    dist_p1 = np.sqrt((p[0]-points[s][0][0])**2+(p[1]-points[s][0][1])**2)
    dist_p2 = np.sqrt((p[0]-points[s][1][0])**2+(p[1]-points[s][1][1])**2)
    dist_p3 = np.sqrt((p[0]-points[s][2][0])**2+(p[1]-points[s][2][1])**2)
    dist_now=np.max([dist_p1,dist_p2,dist_p3])
    dist.append(dist_now)
#    print(cx,cy,dist_p3)
cx=np.array(cx)
cy=np.array(cy)

#mid_points=np.zeros((len(cx),2))

#mid_points=unique2d(mid_points)


mid_points_WCS=np.zeros((len(cx),2))
mid_points_WCS[:,0]=nx/2-cx/(wcs.wcs.cdelt[0]*3600)
mid_points_WCS[:,1]=ny/2+cy/(wcs.wcs.cdelt[1]*3600)

mid_points_WCS=unique2d(mid_points_WCS)

print(len(cx),len(mid_points_WCS))
#
# We clean those points that are too nearby!
#

dist=np.array(dist)
mean_dist=np.around(np.mean(dist),1)
print(mean_dist)



# In[18]:


plt.scatter(points_WCS[:,0],points_WCS[:,1])
plt.scatter(mid_points_WCS[:,0],mid_points_WCS[:,1])


# In[19]:


org_dist_arc=2.7
new_radius=(mean_dist-0.5*org_dist_arc)/pix_size
#new_radius=0.5*mean_dist/pix_size
print('Org radius=',radius,radius*pix_size)
print('new_radius=',new_radius,new_radius*pix_size)
mid_radius=0.5*(mean_dist/pix_size-2*new_radius)
print('mid_radius=',mid_radius,mid_radius*pix_size)
# Final radius:
new_radius = np.round(0.5*(new_radius+mid_radius),2)
mid_radius = np.round(new_radius,2)
print(mid_radius,mid_radius*pix_size)
new_radius=1.0
mid_radius=1.0


# In[20]:


#
# We do new apertures and estimate the values in the center!
#

#new_radius=0.5*mean_dist/pix_size
#mid_radius=2*new_radius-radius
#print('mid_radius=',mid_radius,radius,mid_radius)
#print(new_radius,radius)
aper_center = CircularAperture(points_WCS, new_radius)
aper_ring = CircularAnnulus(points_WCS,new_radius,radius)
phot_tabl = aperture_photometry(hdu.data, aper,                                           method='subpixel',subpixels=5)

phot_table_center_in = aperture_photometry(hdu.data, aper_center,                                           method='subpixel',subpixels=5)
phot_table_out = aperture_photometry(img_map, aper, method='subpixel',subpixels=5)

phot_table_ring_out = aperture_photometry(img_map,                                           aper_ring, method='subpixel',subpixels=5)
phot_table_center_est = phot_table_center_in.copy()
ratio_in_out = phot_table['aperture_sum']/phot_table_out['aperture_sum']
print('mean_ratio=',np.nanmedian(ratio_in_out),np.nanstd(ratio_in_out))
phot_table_center_est['aperture_sum']=phot_table['aperture_sum']-phot_table_ring_out['aperture_sum']*ratio_in_out
print(phot_table_center_est['aperture_sum'][0],phot_table['aperture_sum'][0],phot_table_ring_out['aperture_sum'][0])
delta=phot_table_center_in['aperture_sum']-phot_table_center_est['aperture_sum']
mean_delta=np.nanmean(delta/phot_table_center_in['aperture_sum'])
std_delta=np.nanstd(delta/phot_table_center_in['aperture_sum'])
print(mean_delta,std_delta)

#
# New points!
#
aper_mid_points = CircularAperture(mid_points_WCS, new_radius)
phot_table_mid_points = aperture_photometry(img, aper_mid_points,                                            method='subpixel',subpixels=5)

phot_table_center_est.add_column(new_radius,name='radius')
phot_table_mid_points.add_column(mid_radius,name='radius')

phot_table_final=vstack_table([phot_table_center_est, phot_table_mid_points])
aper_all = vstack_table([aper_center,aper_mid_points])
#print(aper_mid_points)



#plt.scatter(phot_table_center_in['aperture_sum'],phot_table_center_est['aperture_sum'])


# In[21]:


#print(aper_all)


# In[22]:


#print(phot_table_center_est['radius'],radius)


# In[23]:


print(len(phot_table_final))


# In[24]:


def plot_four(img_in,phot_table,img,img_loop):
    fig,ax = plt.subplots(2,2,figsize=(10,8))
    plt.subplot(ax[0][0],projection=wcs)
    ax[0][0].imshow(np.log10(hdu.data), cmap='gray_r', vmin=-3, vmax=0.15, origin='lower')
    plt.subplot(ax[1][0],projection=wcs)
    ax[0][1].imshow(np.log10(hdu.data), cmap='gray_r', vmin=-3, vmax=-3, origin='lower')
    cm = plt.cm.get_cmap('Greys')
    ax[0][1].scatter(phot_table['xcenter'],phot_table['ycenter'],                     c=np.log10(phot_table['aperture_sum']/(np.pi*radius**2)),                     cmap=cm, vmin=-3, vmax=0.15,                     s=140)
    plt.subplot(ax[1][0],projection=wcs)
    ax[1][0].imshow(np.log10(img), cmap='gray_r', vmin=-3, vmax=0.15, origin='lower')

    plt.subplot(ax[1][1],projection=wcs)
    ax[1][1].imshow(np.log10(img_loop), cmap='gray_r', vmin=-3, vmax=0.15, origin='lower')


    ax[0][0].set_xlim([0.2*nx,0.8*nx])
    ax[0][0].set_ylim([0.2*ny,0.8*ny])

    ax[0][1].set_xlim([0.2*nx,0.8*nx])
    ax[0][1].set_ylim([0.2*ny,0.8*ny])

    ax[1][0].set_xlim([0.2*nx,0.8*nx])
    ax[1][0].set_ylim([0.2*ny,0.8*ny])

    ax[1][1].set_xlim([0.2*nx,0.8*nx])
    ax[1][1].set_ylim([0.2*ny,0.8*ny])
    
    


# In[25]:


# New Loop

slope=0.75
img_loop,weights_loop = inter_map_IDW(phot_table_center_est,new_radius,[0,0],[1,1],[480,480],slope=slope,                            radius_limit=0.5*5/pix_size)
print('First Interpolation')
# New image
img_all,weights_loop = inter_map_IDW(phot_table_final,phot_table_final['radius'],                                     [0,0],[1,1],[480,480],slope=slope,                            radius_limit=0.5*5/pix_size)
print('Second Interpolation')

print('Interpolation Done')
img_map = inter_map(phot_table_center_est,new_radius,[0,0],[1,1],[480,480],method='cubic')
plot_four(hdu.data,phot_table,img_loop,img_all)


# In[98]:


slope=2.5

#amplitude=1, x_mean=0, y_mean=0, x_stddev=None, y_stddev=None, theta=None, cov_matrix=None, **kwargs
def Gaussian2D_map(nx,ny,A,xc,yc,sigma_x,sigma_y,theta=0):
    xi = np.arange(0,nx)
    yi = np.arange(0,ny)
    xi,yi = np.meshgrid(xi,yi)    
    img = Gaussian2D(A, xc, yc, sigma_x, sigma_y, theta=theta)(xi, yi)
    return img

#
#
#
FWHM_input = 1.0 # Arcsec
sigma_FWHM=FWHM_input/pix_size/2.354
print('FWHM_input = ', FWHM_input)
img_Gauss=Gaussian2D_map(480,480,1,480*0.5,480*0.5,sigma_FWHM,sigma_FWHM,0)

phot_Gauss = aperture_photometry(img_Gauss, aper_center,                                           method='subpixel',subpixels=5)
img_Gauss_r,weights_Gauss_r = inter_map_IDW(phot_Gauss,new_radius,[0,0],[1,1],[480,480],slope=slope,                            radius_limit=0.5*5/pix_size)

phot_Gauss_org = aperture_photometry(img_Gauss, aper,                                           method='subpixel',subpixels=5)
img_Gauss_org_r,weights_Gauss_r = inter_map_IDW(phot_Gauss_org,radius,[0,0],[1,1],[480,480],slope=slope,                            radius_limit=0.5*5/pix_size)

phot_Gauss_mid = aperture_photometry(img_Gauss, aper_mid_points,                                           method='subpixel',subpixels=5)

phot_Gauss_all=vstack_table([phot_Gauss, phot_Gauss_mid])


img_Gauss_all,weights_Gauss_all = inter_map_IDW(phot_Gauss_all,new_radius,[0,0],[1,1],[480,480],slope=slope,                            radius_limit=0.5*5/pix_size)



fig,ax = plt.subplots(1,4,figsize=(10,8))
ax[0].imshow(np.log10(img_Gauss), cmap='gray_r', vmin=-10, vmax=0.01, origin='lower')
ax[0].set_xlim([0.45*nx,0.55*nx])
ax[0].set_ylim([0.45*ny,0.55*ny])
ax[1].imshow(np.log10(img_Gauss_r), cmap='gray_r', vmin=-10, vmax=0.01, origin='lower')
ax[1].set_xlim([0.45*nx,0.55*nx])
ax[1].set_ylim([0.45*ny,0.55*ny])
ax[2].imshow(np.log10(img_Gauss_org_r), cmap='gray_r', vmin=-10, vmax=0.01, origin='lower')
ax[2].set_xlim([0.45*nx,0.55*nx])
ax[2].set_ylim([0.45*ny,0.55*ny])
ax[3].imshow(np.log10(img_Gauss_all), cmap='gray_r', vmin=-10, vmax=0.01, origin='lower')
ax[3].set_xlim([0.45*nx,0.55*nx])
ax[3].set_ylim([0.45*ny,0.55*ny])

plt.show()


img_psf = img_Gauss_r[int(0.4*ny):int(0.6*ny),int(0.4*nx):int(0.6*nx)]

#img_Gauss_dec = restoration.richardson_lucy(astro_noisy, psf, num_iter=30)


# In[69]:


img_psf_r = img_Gauss_org_r[int(0.4*ny):int(0.6*ny),int(0.4*nx):int(0.6*nx)]
#img_psf_r = img_psf_r/np.sum(img_psf_r)
#plt.imshow(img_psf)
img_Gauss_r += img_Gauss_r.max() * 1E-5 * np.random.random(img_Gauss_r.shape)
#img_Gauss_dec = richardson_lucy_mod(img_Gauss_r, img_psf_r,5)
img_Gauss_dec = restoration.richardson_lucy(img_Gauss_r, img_psf_r,5,filter_epsilon=0.01)
#img_Gauss_dec, chain = restoration.unsupervised_wiener(img_Gauss_org_r, img_psf_r)#, num_iter=30)
#img_Gauss_dec = restoration.wiener(img_Gauss_r, img_psf_r,0.1)#, num_iter=30)
img_Gauss_dec_org = img_Gauss_dec.copy()
img_Gauss_dec = gaussian_filter(img_Gauss_dec,sigma=sigma_FWHM)#*np.sum(img_Gauss)
fig,ax = plt.subplots(1,3,figsize=(10,8))
ax[0].imshow(np.log10(img_Gauss), cmap='gray_r', vmin=-10, vmax=5, origin='lower')
ax[0].set_xlim([0.3*nx,0.7*nx])
ax[0].set_ylim([0.3*ny,0.7*ny])
ax[1].imshow(np.log10(img_Gauss_r), cmap='gray_r', vmin=-10, vmax=5, origin='lower')
ax[1].set_xlim([0.3*nx,0.7*nx])
ax[1].set_ylim([0.3*ny,0.7*ny])
ax[2].imshow(np.log10(img_Gauss_dec), cmap='gray_r', vmin=-10, vmax=5, origin='lower')
ax[2].set_xlim([0.3*nx,0.7*nx])
ax[2].set_ylim([0.3*ny,0.7*ny])

plt.show()


# In[70]:


def measure_fwhm(array):
    """Fit a Gaussian2D model to a PSF and return the FWHM

    Parameters
    ----------
    array : numpy.ndarray
        Array containing PSF

    Returns
    -------
    x_fwhm : float
        FWHM in x direction in units of pixels

    y_fwhm : float
        FWHM in y direction in units of pixels
    """
    yp, xp = array.shape
    y, x, = np.mgrid[:yp, :xp]
    p_init = models.Gaussian2D(10,xp/2,yp/2)
    fit_p = fitting.LevMarLSQFitter()
    fitted_psf = fit_p(p_init, x, y, array)
    #print(fitted_psf)
    model_out = models.Gaussian2D(fitted_psf.amplitude,fitted_psf.x_mean,fitted_psf.y_mean,                                fitted_psf.x_stddev,fitted_psf.y_stddev,fitted_psf.theta)
    img_out = model_out(y,x)
    #print(img_out)
    fig,ax = plt.subplots(1,2,figsize=(8,6))
    ax[0].imshow(np.log10(array), cmap='gray_r', vmin=-10, vmax=5, origin='lower')
    ax[1].imshow(np.log10(img_out), cmap='gray_r', vmin=-10, vmax=5, origin='lower')
    return fitted_psf

fitted_psf = measure_fwhm(img_Gauss[int(0.3*ny):int(0.7*ny),int(0.3*nx):int(0.7*nx)])
fwhm_out=np.sqrt(fitted_psf.y_fwhm**2+fitted_psf.x_fwhm**2)*pix_size/np.sqrt(2)
print('FWHM_out = ',fwhm_out)
#img_Gauss_dec

fitted_psf = measure_fwhm(img_Gauss_dec[int(0.3*ny):int(0.7*ny),int(0.3*nx):int(0.7*nx)])
fwhm_out=np.sqrt(fitted_psf.y_fwhm**2+fitted_psf.x_fwhm**2)*pix_size/np.sqrt(2)
print('FWHM_out_dec = ',fwhm_out)

#print(fitted_psf.amplitude,fitted_psf.x_mean,fitted_psf.y_mean,\
#      fitted_psf.x_stddev,fitted_psf.y_stddev,fitted_psf.theta,fitted_psf.y_fwhm,fitted_psf.x_fwhm)


# In[79]:


a_FWHM_input = np.arange(0.7,2.0,0.1)
a_FWHM_in=[]
a_FWHM_out=[]


# In[80]:


#a_FWHM_input = np.arange(0.7,1.7,0.1)
for FWHM_input in a_FWHM_input:
    sigma_FWHM=FWHM_input/pix_size/2.354
    print('FWHM_input = ', FWHM_input)
    img_Gauss=Gaussian2D_map(480,480,1,480*0.5,480*0.5,sigma_FWHM,sigma_FWHM,0)
    phot_Gauss = aperture_photometry(img_Gauss, aper_center,                                     method='subpixel',subpixels=5)
    img_Gauss_r,weights_Gauss_r = inter_map_IDW(phot_Gauss,new_radius,[0,0],[1,1],[480,480],slope=slope,                                                radius_limit=0.5*5/pix_size)

    phot_Gauss_org = aperture_photometry(img_Gauss, aper,                                         method='subpixel',subpixels=5)
    img_Gauss_org_r,weights_Gauss_r = inter_map_IDW(phot_Gauss_org,radius,[0,0],[1,1],[480,480],slope=slope,                                                    radius_limit=0.5*5/pix_size)

    phot_Gauss_mid = aperture_photometry(img_Gauss, aper_mid_points,                                         method='subpixel',subpixels=5)
    
    phot_Gauss_all=vstack_table([phot_Gauss, phot_Gauss_mid])


    img_Gauss_all,weights_Gauss_all = inter_map_IDW(phot_Gauss_all,new_radius,[0,0],[1,1],[480,480],slope=slope,                                                    radius_limit=0.5*5/pix_size)


    img_psf = img_Gauss_r[int(0.4*ny):int(0.6*ny),int(0.4*nx):int(0.6*nx)]
    img_psf_all_r = img_Gauss_all[int(0.4*ny):int(0.6*ny),int(0.4*nx):int(0.6*nx)]

    img_Gauss_all += img_Gauss_all.max() * 1E-5 * np.random.random(img_Gauss_all.shape)

    img_Gauss_dec, chain = restoration.unsupervised_wiener(img_Gauss_all, img_psf_all_r)#, num_iter=30)

    img_Gauss_dec_org = img_Gauss_dec.copy()
    img_Gauss_dec = gaussian_filter(img_Gauss_dec,sigma=sigma_FWHM)#*np.sum(img_Gauss)
    fitted_psf = measure_fwhm(img_Gauss[int(0.3*ny):int(0.7*ny),int(0.3*nx):int(0.7*nx)])
    fwhm_out=np.sqrt(fitted_psf.y_fwhm**2+fitted_psf.x_fwhm**2)*pix_size/np.sqrt(2)
    print('FWHM_out = ',fwhm_out)
    #img_Gauss_dec
    a_FWHM_in.append(fwhm_out)
    fitted_psf = measure_fwhm(img_Gauss_dec[int(0.3*ny):int(0.7*ny),int(0.3*nx):int(0.7*nx)])
    fwhm_out=np.sqrt(fitted_psf.y_fwhm**2+fitted_psf.x_fwhm**2)*pix_size/np.sqrt(2)
    print('FWHM_out_dec = ',fwhm_out)
    a_FWHM_out.append(fwhm_out)

a_FWHM_in=np.array(a_FWHM_in)
a_FWHM_out=np.array(a_FWHM_out)
#img_Gauss_dec = restoration.richardson_lucy(astro_noisy, psf, num_iter=30)


# In[81]:


a_FWHM_in_ALL = a_FWHM_in.copy()
a_FWHM_out_ALL = a_FWHM_out.copy()


# For a FWHM_input=0.7"  => FWHM_out=1.3"
# For a FWHM_input=0.85" => FWHM_out=1.4"
# For a FWHM_input=1.0"  => FWHM_out=1.5"
# For a FWHM_input=
# For a FWHM_input=1.5"  => FWHM_out=2.0"
#


# In[85]:


plt.scatter(a_FWHM_input,a_FWHM_in_ALL)
plt.scatter(a_FWHM_input,a_FWHM_out_ALL)


# In[92]:


#a_FWHM_input = np.arange(0.7,1.7,0.1)
a_FWHM_input = np.arange(0.7,2.0,0.1)
#a_FWHM_input = np.arange(1.0,1.2,0.1)
a_FWHM_in=[]
a_FWHM_out=[]

f_e=0.01
for FWHM_input in a_FWHM_input:
    sigma_FWHM=FWHM_input/pix_size/2.354
    print('FWHM_input = ', FWHM_input)
    img_Gauss=Gaussian2D_map(480,480,1,480*0.5,480*0.5,sigma_FWHM,sigma_FWHM,0)
    phot_Gauss = aperture_photometry(img_Gauss, aper_center,                                     method='subpixel',subpixels=5)
    img_Gauss_r,weights_Gauss_r = inter_map_IDW(phot_Gauss,new_radius,[0,0],[1,1],[480,480],slope=slope,                                                radius_limit=0.5*5/pix_size)
    img_psf = img_Gauss_r[int(0.4*ny):int(0.6*ny),int(0.4*nx):int(0.6*nx)]
    img_Gauss_r += img_Gauss_r.max() * 1E-5 * np.random.random(img_Gauss_r.shape)
    img_Gauss_dec=restoration.richardson_lucy(img_Gauss_r, img_psf_r,10,clip=True,filter_epsilon=f_e)
    img_Gauss_dec_org = img_Gauss_dec.copy()
    img_Gauss_dec = gaussian_filter(img_Gauss_dec,sigma=sigma_FWHM)#*np.sum(img_Gauss)
    fitted_psf = measure_fwhm(img_Gauss[int(0.3*ny):int(0.7*ny),int(0.3*nx):int(0.7*nx)])
    fwhm_out=np.sqrt(fitted_psf.y_fwhm**2+fitted_psf.x_fwhm**2)*pix_size/np.sqrt(2)
    print('FWHM_out = ',fwhm_out)
    #img_Gauss_dec
    a_FWHM_in.append(fwhm_out)
    fitted_psf = measure_fwhm(img_Gauss_dec[int(0.3*ny):int(0.7*ny),int(0.3*nx):int(0.7*nx)])
    fwhm_out=np.sqrt(fitted_psf.y_fwhm**2+fitted_psf.x_fwhm**2)*pix_size/np.sqrt(2)
    print('FWHM_out_dec = ',fwhm_out)
    a_FWHM_out.append(fwhm_out)

a_FWHM_in=np.array(a_FWHM_in)
a_FWHM_out=np.array(a_FWHM_out)
#img_Gauss_dec = restoration.richardson_lucy(astro_noisy, psf, num_iter=30)


# In[94]:


a_FWHM_in_r = a_FWHM_in.copy()
a_FWHM_out_r = a_FWHM_out.copy()


# In[95]:


plt.scatter(a_FWHM_input,a_FWHM_in_r)
plt.scatter(a_FWHM_input,a_FWHM_out_r)


# In[93]:


FWHM_input=1.0
sigma_FWHM=FWHM_input/pix_size/2.354
print('FWHM_input = ', FWHM_input)
img_Gauss=Gaussian2D_map(480,480,1,480*0.5,480*0.5,sigma_FWHM,sigma_FWHM,0)
phot_Gauss = aperture_photometry(img_Gauss, aper_center,                                 method='subpixel',subpixels=5)
img_Gauss_r,weights_Gauss_r = inter_map_IDW(phot_Gauss,new_radius,[0,0],[1,1],[480,480],slope=slope,                                            radius_limit=0.5*5/pix_size)
img_psf = img_Gauss_r[int(0.4*ny):int(0.6*ny),int(0.4*nx):int(0.6*nx)]


# In[126]:


img_loop = img_center
img_psf_r = img_Gauss_r[int(0.4*ny):int(0.6*ny),int(0.4*nx):int(0.6*nx)]
#img_psf_r = img_psf_r/np.sum(img_psf_r)
#img_loop_in = img_loop+img_loop.max() * 1E-5 * np.random.random(img_loop.shape)
img_loop_in = img_center+img_center.max() * 1E-5 * np.random.random(img_center.shape)
img_loop_dec = restoration.richardson_lucy(img_loop_in, img_psf_r,10)#, num_iter=30)
#img_loop_dec, chain = restoration.unsupervised_wiener(img_loop, img_psf_r)#, num_iter=30)
#img_loop_dec = restoration.wiener(img_loop, img_psf_r,100000)#, num_iter=30)
img_loop_dec = gaussian_filter(img_loop_dec,sigma=1.0/pix_size/2.354)#*np.sum(img_Gauss)
#img_loop_dec = img_loop_de
#img_loop_dec=convolve2d(img_loop_dec, img_Gauss_dec_org[int(0.45*ny):int(0.55*ny),int(0.45*nx):int(0.55*nx)])[0:ny,0:nx]
flux_ratio = np.sum(img_loop)/np.sum(img_loop_dec)
img_loop_dec = img_loop_dec*flux_ratio
img_loop_dec[img_loop==0]=0


fig,ax = plt.subplots(1,3,figsize=(10,8))
ax[0].imshow(np.log10(hdu.data), cmap='magma_r', vmin=-3, vmax=0.15, origin='lower')
ax[0].set_xlim([0.2*nx,0.8*nx])
ax[0].set_ylim([0.2*ny,0.8*ny])
ax[1].imshow(np.log10(img_center), cmap='magma_r', vmin=-3, vmax=0.15, origin='lower')
ax[1].set_xlim([0.2*nx,0.8*nx])
ax[1].set_ylim([0.2*ny,0.8*ny])
ax[2].imshow(np.log10(img_loop_dec), cmap='magma_r', vmin=-3, vmax=0.15, origin='lower')
ax[2].set_xlim([0.2*nx,0.8*nx])
ax[2].set_ylim([0.2*ny,0.8*ny])

img_loop_center_dec=img_loop_dec
plt.show()


# In[127]:


img_test=img_loop_dec
#plot_three(hdu.data,img_test,hdu.data-img_test)
diff_min=1e12
sigma_min=0.1
for sigma in arange(0.2,10,0.1):
    s_img = gaussian_filter(hdu.data,sigma=sigma)
    diff = img_test-s_img
    d=img_test.shape
    mean_diff = np.sum(np.abs(diff[img_test>0]/img_test[img_test>0])**2)/(d[0]*d[1])
    #print(sigma,mean_diff)
    if (mean_diff<diff_min):
        diff_min=mean_diff
        sigma_min=sigma
FWHM_PS = 1.0
print('sigma_min=',sigma_min,sigma_min*pix_size)
print('Final FWHM = ',np.sqrt(FWHM_PS**2+sigma_min*pix_size*2.354))
s_img = gaussian_filter(hdu.data,sigma=sigma_min)
plot_three(s_img,img_test,s_img-img_test)


# In[123]:


#
# FURTHER TESTS 3.5.2022
#


# In[ ]:


#sigma_min= 3.4000000000000012 0.8500000205822257
#Final FWHM =  1.7323106096917376


# In[99]:


img_psf_all_r = img_Gauss_all[int(0.4*ny):int(0.6*ny),int(0.4*nx):int(0.6*nx)]
#img_psf_r = img_psf_r/np.sum(img_psf_r)
#plt.imshow(img_psf)
img_Gauss_all += img_Gauss_all.max() * 1E-5 * np.random.random(img_Gauss_all.shape)
#img_Gauss_dec = restoration.richardson_lucy(img_Gauss_r, img_psf_r,5)
img_Gauss_dec, chain = restoration.unsupervised_wiener(img_Gauss_all, img_psf_all_r)#, num_iter=30)
#img_Gauss_dec = restoration.wiener(img_Gauss_r, img_psf_r,0.1)#, num_iter=30)
img_Gauss_dec_org = img_Gauss_dec.copy()
img_Gauss_dec = gaussian_filter(img_Gauss_dec,sigma=sigma_FWHM)#*np.sum(img_Gauss)
fig,ax = plt.subplots(1,3,figsize=(10,8))
ax[0].imshow(np.log10(img_Gauss), cmap='gray_r', vmin=-10, vmax=5, origin='lower')
ax[0].set_xlim([0.3*nx,0.7*nx])
ax[0].set_ylim([0.3*ny,0.7*ny])
ax[1].imshow(np.log10(img_Gauss_all), cmap='gray_r', vmin=-10, vmax=5, origin='lower')
ax[1].set_xlim([0.3*nx,0.7*nx])
ax[1].set_ylim([0.3*ny,0.7*ny])
ax[2].imshow(np.log10(img_Gauss_dec), cmap='gray_r', vmin=-10, vmax=5, origin='lower')
ax[2].set_xlim([0.3*nx,0.7*nx])
ax[2].set_ylim([0.3*ny,0.7*ny])

plt.show()


# In[ ]:


slope=0.75
img_org_075,weights_org = inter_map_IDW(phot_table,                                      radius,[0,0],[1,1],[480,480],slope=slope,                            radius_limit=3.5/pix_size)


# In[ ]:


img_test=img_org_075
#plot_three(hdu.data,img_test,hdu.data-img_test)
diff_min=1e12
sigma_min=0.1
for sigma in arange(0.2,10,0.1):
    s_img = gaussian_filter(hdu.data,sigma=sigma)
    diff = img_test-s_img
    d=img_test.shape
    mean_diff = np.sum(np.abs(diff[img_test>0]/img_test[img_test>0])**2)/(d[0]*d[1])
    #print(sigma,mean_diff)
    if (mean_diff<diff_min):
        diff_min=mean_diff
        sigma_min=sigma
FWHM_PS = 1.0
print('sigma_min=',sigma_min,sigma_min*pix_size)
print('Final FWHM = ',np.sqrt(FWHM_PS**2+sigma_min*pix_size*2.354))
s_img = gaussian_filter(hdu.data,sigma=sigma_min)
plot_three(s_img,img_test,s_img-img_test)


# In[ ]:


slope=2.5
img_org,weights_org = inter_map_IDW(phot_table,                                      radius,[0,0],[1,1],[480,480],slope=slope,                            radius_limit=3.5/pix_size)
img_loop=img_org


# In[100]:


img_test=img_org
#plot_three(hdu.data,img_test,hdu.data-img_test)
diff_min=1e12
sigma_min=0.1
for sigma in arange(0.2,10,0.1):
    s_img = gaussian_filter(hdu.data,sigma=sigma)
    diff = img_test-s_img
    d=img_test.shape
    mean_diff = np.sum(np.abs(diff[img_test>0]/img_test[img_test>0])**2)/(d[0]*d[1])
    #print(sigma,mean_diff)
    if (mean_diff<diff_min):
        diff_min=mean_diff
        sigma_min=sigma
FWHM_PS = 1.0
print('sigma_min=',sigma_min,sigma_min*pix_size)
print('Final FWHM = ',np.sqrt(FWHM_PS**2+sigma_min*pix_size*2.354))
s_img = gaussian_filter(hdu.data,sigma=sigma_min)
plot_three(s_img,img_test,s_img-img_test)


# In[103]:


slope=2.5
img_center,weights_center = inter_map_IDW(phot_table_center_est,                                      new_radius,[0,0],[1,1],[480,480],slope=slope,                            radius_limit=3.5/pix_size)


# In[104]:


#print(phot_table_center_est)


# In[105]:


img_test=img_center
#plot_three(hdu.data,img_test,hdu.data-img_test)
diff_min=1e12
sigma_min=0.1
for sigma in arange(0.2,10,0.1):
    s_img = gaussian_filter(hdu.data,sigma=sigma)
    diff = img_test-s_img
    d=img_test.shape
    mean_diff = np.sum(np.abs(diff[img_test>0]/img_test[img_test>0])**2)/(d[0]*d[1])
    #print(sigma,mean_diff)
    if (mean_diff<diff_min):
        diff_min=mean_diff
        sigma_min=sigma
FWHM_PS = 1.0
print('sigma_min=',sigma_min,sigma_min*pix_size)
print('Final FWHM = ',np.sqrt(FWHM_PS**2+sigma_min*pix_size*2.354))
s_img = gaussian_filter(hdu.data,sigma=sigma_min)
plot_three(s_img,img_test,s_img-img_test)


# In[120]:


img_loop = img_center
img_psf_r = img_Gauss_r[int(0.4*ny):int(0.6*ny),int(0.4*nx):int(0.6*nx)]
#img_psf_r = img_psf_r/np.sum(img_psf_r)
#img_loop_in = img_loop+img_loop.max() * 1E-5 * np.random.random(img_loop.shape)
img_loop_in = img_center+img_center.max() * 1E-5 * np.random.random(img_center.shape)
img_loop_dec = restoration.richardson_lucy(img_loop_in, img_psf_r,10)#, num_iter=30)
#img_loop_dec, chain = restoration.unsupervised_wiener(img_loop, img_psf_r)#, num_iter=30)
#img_loop_dec = restoration.wiener(img_loop, img_psf_r,100000)#, num_iter=30)
img_loop_dec = gaussian_filter(img_loop_dec,sigma=1.25/pix_size/2.354)#*np.sum(img_Gauss)
#img_loop_dec = img_loop_de
#img_loop_dec=convolve2d(img_loop_dec, img_Gauss_dec_org[int(0.45*ny):int(0.55*ny),int(0.45*nx):int(0.55*nx)])[0:ny,0:nx]
flux_ratio = np.sum(img_loop)/np.sum(img_loop_dec)
img_loop_dec = img_loop_dec*flux_ratio
img_loop_dec[img_loop==0]=0


fig,ax = plt.subplots(1,3,figsize=(10,8))
ax[0].imshow(np.log10(hdu.data), cmap='magma_r', vmin=-3, vmax=0.15, origin='lower')
ax[0].set_xlim([0.2*nx,0.8*nx])
ax[0].set_ylim([0.2*ny,0.8*ny])
ax[1].imshow(np.log10(img_center), cmap='magma_r', vmin=-3, vmax=0.15, origin='lower')
ax[1].set_xlim([0.2*nx,0.8*nx])
ax[1].set_ylim([0.2*ny,0.8*ny])
ax[2].imshow(np.log10(img_loop_dec), cmap='magma_r', vmin=-3, vmax=0.15, origin='lower')
ax[2].set_xlim([0.2*nx,0.8*nx])
ax[2].set_ylim([0.2*ny,0.8*ny])

img_loop_center_dec=img_loop_dec
plt.show()


# In[121]:


img_test=img_loop_dec
#plot_three(hdu.data,img_test,hdu.data-img_test)
diff_min=1e12
sigma_min=0.1
for sigma in arange(0.2,10,0.1):
    s_img = gaussian_filter(hdu.data,sigma=sigma)
    diff = img_test-s_img
    d=img_test.shape
    mean_diff = np.sum(np.abs(diff[img_test>0]/img_test[img_test>0])**2)/(d[0]*d[1])
    #print(sigma,mean_diff)
    if (mean_diff<diff_min):
        diff_min=mean_diff
        sigma_min=sigma
FWHM_PS = 1.0
print('sigma_min=',sigma_min,sigma_min*pix_size)
print('Final FWHM = ',np.sqrt(FWHM_PS**2+sigma_min*pix_size*2.354))
s_img = gaussian_filter(hdu.data,sigma=sigma_min)
plot_three(s_img,img_test,s_img-img_test)


# In[ ]:


img_psf_r = img_Gauss_all[int(0.4*ny):int(0.6*ny),int(0.4*nx):int(0.6*nx)]
#img_psf_r = img_psf_r/np.sum(img_psf_r)
#img_loop_in = img_loop+img_loop.max() * 1E-5 * np.random.random(img_loop.shape)
img_loop_in = img_all+img_all.max() * 1E-5 * np.random.random(img_all.shape)
img_loop_dec = restoration.richardson_lucy(img_loop_in, img_psf_r,10)#, num_iter=30)
#img_loop_dec, chain = restoration.unsupervised_wiener(img_loop, img_psf_r)#, num_iter=30)
#img_loop_dec = restoration.wiener(img_loop, img_psf_r,100000)#, num_iter=30)
img_loop_dec = gaussian_filter(img_loop_dec,sigma=1.5)#*np.sum(img_Gauss)
#img_loop_dec = img_loop_de
#img_loop_dec=convolve2d(img_loop_dec, img_Gauss_dec_org[int(0.45*ny):int(0.55*ny),int(0.45*nx):int(0.55*nx)])[0:ny,0:nx]
flux_ratio = np.sum(img_loop)/np.sum(img_loop_dec)
img_loop_dec = img_loop_dec*flux_ratio
img_loop_dec[img_loop==0]=0


fig,ax = plt.subplots(1,3,figsize=(10,8))
ax[0].imshow(np.log10(hdu.data), cmap='magma_r', vmin=-3, vmax=0.15, origin='lower')
ax[0].set_xlim([0.2*nx,0.8*nx])
ax[0].set_ylim([0.2*ny,0.8*ny])
ax[1].imshow(np.log10(img_all), cmap='magma_r', vmin=-3, vmax=0.15, origin='lower')
ax[1].set_xlim([0.2*nx,0.8*nx])
ax[1].set_ylim([0.2*ny,0.8*ny])
ax[2].imshow(np.log10(img_loop_dec), cmap='magma_r', vmin=-3, vmax=0.15, origin='lower')
ax[2].set_xlim([0.2*nx,0.8*nx])
ax[2].set_ylim([0.2*ny,0.8*ny])

plt.show()


# In[ ]:


img_psf_r = img_Gauss_org_r[int(0.4*ny):int(0.6*ny),int(0.4*nx):int(0.6*nx)]
#img_psf_r = img_psf_r/np.sum(img_psf_r)
#img_loop_in = img_loop+img_loop.max() * 1E-5 * np.random.random(img_loop.shape)
img_loop_in = img_org+img_org.max() * 1E-5 * np.random.random(img_org.shape)
img_loop_dec = restoration.richardson_lucy(img_loop_in, img_psf_r,10)#, num_iter=30)
#img_loop_dec, chain = restoration.unsupervised_wiener(img_loop, img_psf_r)#, num_iter=30)
#img_loop_dec = restoration.wiener(img_loop, img_psf_r,100000)#, num_iter=30)
img_loop_dec = gaussian_filter(img_loop_dec,sigma=1.5)#*np.sum(img_Gauss)
#img_loop_dec = img_loop_de
#img_loop_dec=convolve2d(img_loop_dec, img_Gauss_dec_org[int(0.45*ny):int(0.55*ny),int(0.45*nx):int(0.55*nx)])[0:ny,0:nx]
flux_ratio = np.sum(img_loop)/np.sum(img_loop_dec)
img_loop_dec = img_loop_dec*flux_ratio
img_loop_dec[img_loop==0]=0


fig,ax = plt.subplots(1,3,figsize=(10,8))
ax[0].imshow(np.log10(hdu.data), cmap='magma_r', vmin=-3, vmax=0.15, origin='lower')
ax[0].set_xlim([0.2*nx,0.8*nx])
ax[0].set_ylim([0.2*ny,0.8*ny])
ax[1].imshow(np.log10(img_org), cmap='magma_r', vmin=-3, vmax=0.15, origin='lower')
ax[1].set_xlim([0.2*nx,0.8*nx])
ax[1].set_ylim([0.2*ny,0.8*ny])
ax[2].imshow(np.log10(img_loop_dec), cmap='magma_r', vmin=-3, vmax=0.15, origin='lower')
ax[2].set_xlim([0.2*nx,0.8*nx])
ax[2].set_ylim([0.2*ny,0.8*ny])

plt.show()


# In[ ]:


img_test=img_loop_dec
#plot_three(hdu.data,img_test,hdu.data-img_test)
diff_min=1e12
sigma_min=0.1
for sigma in arange(0.2,10,0.1):
    s_img = gaussian_filter(hdu.data,sigma=sigma)
    diff = img_test-s_img
    d=img_test.shape
    mean_diff = np.sum(np.abs(diff[img_test>0]/img_test[img_test>0])**2)/(d[0]*d[1])
    #print(sigma,mean_diff)
    if (mean_diff<diff_min):
        diff_min=mean_diff
        sigma_min=sigma
FWHM_PS = 1.0
print('sigma_min=',sigma_min,sigma_min*pix_size)
print('Final FWHM = ',np.sqrt(FWHM_PS**2+sigma_min*pix_size*2.354))
s_img = gaussian_filter(hdu.data,sigma=sigma_min)
plot_three(s_img,img_test,s_img-img_test)


# In[ ]:


flat_img = img_loop_dec/img_org


# In[ ]:


img_test=img_loop_center_dec
#plot_three(hdu.data,img_test,hdu.data-img_test)
diff_min=1e12
sigma_min=0.1
for sigma in arange(0.2,10,0.1):
    s_img = gaussian_filter(hdu.data,sigma=sigma)
    diff = img_test-s_img
    d=img_test.shape
    mean_diff = np.sum(np.abs(diff[img_test>0]/img_test[img_test>0])**2)/(d[0]*d[1])
    #print(sigma,mean_diff)
    if (mean_diff<diff_min):
        diff_min=mean_diff
        sigma_min=sigma
FWHM_PS = 1.0
print('sigma_min=',sigma_min,sigma_min*pix_size)
print('Final FWHM = ',np.sqrt(FWHM_PS**2+sigma_min*pix_size*2.354))
s_img = gaussian_filter(hdu.data,sigma=sigma_min)
plot_three(s_img,img_test,s_img-img_test)


# In[ ]:


img_test=img_loop_dec
#plot_three(hdu.data,img_test,hdu.data-img_test)
diff_min=1e12
sigma_min=0.1
for sigma in arange(0.2,10,0.1):
    s_img = gaussian_filter(hdu.data,sigma=sigma)
    diff = img_test-s_img
    d=img_test.shape
    mean_diff = np.sum(np.abs(diff[img_test>0]/img_test[img_test>0])**2)/(d[0]*d[1])
    #print(sigma,mean_diff)
    if (mean_diff<diff_min):
        diff_min=mean_diff
        sigma_min=sigma
FWHM_PS = 1.0
print('sigma_min=',sigma_min,sigma_min*pix_size)
print('Final FWHM = ',np.sqrt(FWHM_PS**2+sigma_min*pix_size*2.354))
s_img = gaussian_filter(hdu.data,sigma=sigma_min)
plot_three(s_img,img_test,s_img-img_test)


# In[ ]:


#
# Best number of iterations!
#
for niter in arange(1,100,5):
    img_loop_in = img_loop+img_loop.max() * 1E-5 * np.random.random(img_loop.shape)
    img_loop_dec = restoration.richardson_lucy(img_loop_in, img_psf_r,niter)#, num_iter=30)
    img_loop_dec = gaussian_filter(img_loop_dec,sigma=1.5)#*np.sum(img_Gauss)
    flux_ratio = np.sum(img_loop)/np.sum(img_loop_dec)
    img_loop_dec = img_loop_dec*flux_ratio
    img_loop_dec[img_loop==0]=0
    
    img_test=img_loop_dec
    diff_min=1e12
    sigma_min=0.1
    for sigma in arange(0.2,10,0.1):
        s_img = gaussian_filter(hdu.data,sigma=sigma)
        diff = img_test-s_img
        d=img_test.shape
        mean_diff = np.sum(np.abs(diff[img_test>0]/img_test[img_test>0])**2)/(d[0]*d[1])
        std_diff = np.std(diff[(img_test>0) & (np.abs(diff)<0.5)]/img_test[(img_test>0) & (np.abs(diff)<0.5)])
        if (mean_diff<diff_min):
            diff_min=mean_diff
            sigma_min=sigma
            std_min=std_diff
    FWHM_PS = 1.0
    print('Niter=',niter,'sigma_min=',sigma_min,sigma_min*pix_size,          'Final FWHM = ',np.sqrt(FWHM_PS**2+sigma_min*pix_size*2.354),',std=',std_min)
    s_img = gaussian_filter(hdu.data,sigma=sigma_min)
    plot_three(s_img,img_test,s_img-img_test)


# In[ ]:


print(img_Gauss_dec[img_Gauss_dec>0])


# In[ ]:


slope=2.5
img,weights = inter_map_IDW(phot_table,radius,[0,0],[1,1],[480,480],slope=slope,                            radius_limit=5/pix_size)


# In[ ]:





# In[ ]:


img_map_old = inter_map(phot_table,radius,[0,0],[1,1],[480,480],method='cubic')
img_map = inter_map(phot_table,radius,[0,0],[1,1],[480,480],method='cubic')
img_map_new = inter_map(phot_table_final,new_radius,[0,0],[1,1],[480,480],method='linear')
plot_four(hdu.data,phot_table,img_map,img_map_new)


# In[ ]:


slope=0.5
img_loop,weights_loop = inter_map_IDW(phot_table_center_est,new_radius,[0,0],[1,1],[480,480],slope=slope,                            radius_limit=0.5*5/pix_size)


# In[ ]:


#slope=2.5
#img_loop,weights_loop = inter_map_IDW(phot_table_center_est,\
#                                      new_radius,[0,0],[1,1],[480,480],slope=slope,\
#                            radius_limit=5/pix_size)
#print('slope=',slope)
img_test=img_loop_dec
#plot_three(hdu.data,img_test,hdu.data-img_test)
diff_min=1e12
sigma_min=0.1
for sigma in arange(0.2,10,0.1):
    s_img = gaussian_filter(hdu.data,sigma=sigma)
    diff = img_test-s_img
    d=img_test.shape
    mean_diff = np.sum(np.abs(diff[img_test>0]/img_test[img_test>0])**2)/(d[0]*d[1])
    #print(sigma,mean_diff)
    if (mean_diff<diff_min):
        diff_min=mean_diff
        sigma_min=sigma
FWHM_PS = 1.0
print('sigma_min=',sigma_min,sigma_min*pix_size)
print('Final FWHM = ',np.sqrt(FWHM_PS**2+sigma_min*pix_size*2.354))
s_img = gaussian_filter(hdu.data,sigma=sigma_min)
plot_three(s_img,img_test,s_img-img_test)


# In[ ]:


img_test=img_map_new
#plot_three(hdu.data,img_test,hdu.data-img_test)
diff_min=1e12
sigma_min=0.1
for sigma in arange(0.2,5,0.1):
    s_img = gaussian_filter(hdu.data,sigma=sigma)
    diff = img_test-s_img
    mean_diff = np.sum(np.abs(diff[img_all>0.1]))
    if (mean_diff<diff_min):
        diff_min=mean_diff
        sigma_min=sigma
FWHM_PS = 1.0
print('sigma_min=',sigma_min,sigma_min*pix_size)
print('Final FWHM = ',np.sqrt(FWHM_PS**2+sigma_min*pix_size*2.354))
s_img = gaussian_filter(hdu.data,sigma=sigma_min)
plot_three(s_img,img_test,s_img-img_test)


# In[96]:


slope=2.5
img_org,weights_org = inter_map_IDW(phot_table,                                      radius,[0,0],[1,1],[480,480],slope=slope,                            radius_limit=5/pix_size)
print('slope=',slope)
img_test=img_org
#plot_three(hdu.data,img_test,hdu.data-img_test)
diff_min=1e12
sigma_min=0.1
for sigma in arange(0.2,5,0.1):
    s_img = gaussian_filter(hdu.data,sigma=sigma)
    diff = img_test-s_img
    mean_diff = np.sum(np.abs(diff[img_all>0.1]))
    if (mean_diff<diff_min):
        diff_min=mean_diff
        sigma_min=sigma
FWHM_PS = 1.0
print('sigma_min=',sigma_min,sigma_min*pix_size)
print('Final FWHM = ',np.sqrt(FWHM_PS**2+sigma_min*pix_size*2.354))
s_img = gaussian_filter(hdu.data,sigma=sigma_min)
plot_three(s_img,img_test,s_img-img_test)


# In[ ]:


for slope in arange(0.5,5,0.25):
    img_loop,weights_loop = inter_map_IDW(phot_table_center_est,                                          new_radius,[0,0],[1,1],[480,480],slope=slope,                            radius_limit=0.5*5/pix_size)
    print('slope=',slope)
    img_test=img_loop
    #plot_three(hdu.data,img_test,hdu.data-img_test)
    diff_min=1e12
    sigma_min=0.1
    for sigma in arange(0.2,5,0.1):
        s_img = gaussian_filter(hdu.data,sigma=sigma)
        diff = img_test-s_img
        mean_diff = np.sum(np.abs(diff[img_all>0.1]))
        std_diff = np.std()
        if (mean_diff<diff_min):
            diff_min=mean_diff
            sigma_min=sigma
    FWHM_PS = 1.0
    print('sigma_min=',sigma_min,sigma_min*pix_size)
    print('Final FWHM = ',np.sqrt(FWHM_PS**2+sigma_min*pix_size*2.354))
    s_img = gaussian_filter(hdu.data,sigma=sigma_min)
    plot_three(s_img,img_test,s_img-img_test)
    
    


# In[ ]:


kernel = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])

kernel = -(1 / 256.0) * np.array([[1, 4, 6, 4, 1],
                                   [4, 16, 24, 16, 4],
                                   [6, 24, -476, 24, 6],
                                   [4, 16, 24, 16, 4],
                                   [1, 4, 6, 4, 1]])
img_test=img_map_old #convolve2d(img_loop, kernel)[0:ny,0:nx]

#img_test=img_loop #denoise_tv_bregman(img_loop,1)

plot_three(hdu.data,img_test,hdu.data-img_test)
diff_min=1e12
sigma_min=0.1
for sigma in arange(0.2,5,0.1):
    s_img = gaussian_filter(hdu.data,sigma=sigma)
    diff = img_test-s_img
    mean_diff = np.sum(np.abs(diff[img_all>0.1]))
    if (mean_diff<diff_min):
        diff_min=mean_diff
        sigma_min=sigma
FWHM_PS = 1.5
print('sigma_min=',sigma_min,sigma_min*pix_size)
print('Final FWHM = ',np.sqrt(FWHM_PS**2+sigma_min*pix_size*2.354))
s_img = gaussian_filter(hdu.data,sigma=sigma_min)
plot_three(s_img,img_test,s_img-img_test)


# In[ ]:


print('Interpolation Done')
img_map_old = inter_map(phot_table,radius,[0,0],[1,1],[480,480],method='cubic')
img_map = inter_map(phot_table,radius,[0,0],[1,1],[480,480],method='cubic')
img_map_new = inter_map(phot_table_final,new_radius,[0,0],[1,1],[480,480],method='cubic')
plot_four(hdu.data,phot_table,img_map,img_map_new)


# In[97]:


#
# Loop
#
aper_in = CircularAperture(points_WCS, radius)
aper_center = CircularAperture(points_WCS, new_radius)
aper_ring = CircularAnnulus(points_WCS,new_radius,radius)
#img_map,weights_loop = inter_map_IDW(phot_table,radius,[0,0],[1,1],[480,480],slope=1.5,\
#                            radius_limit=0.5*5/pix_size)

#inter_map(phot_table,radius,[0,0],[1,1],[480,480],method='nearest')
phot_table_in = aperture_photometry(img_all, aper_in,                                           method='subpixel',subpixels=5)
phot_table_center_in = aperture_photometry(img_all, aper_center,                                           method='subpixel',subpixels=5)
phot_table_ring_out = aperture_photometry(img_all, aper_ring,                                          method='subpixel',subpixels=5)
phot_table_center_est = phot_table_center_in.copy()
phot_table_center_est['aperture_sum']=phot_table['aperture_sum']-phot_table_ring_out['aperture_sum']
delta=phot_table_center_in['aperture_sum']-phot_table_center_est['aperture_sum']
mean_delta=np.nanmean(delta/phot_table_center_in['aperture_sum'])
std_delta=np.nanstd(delta/phot_table_center_in['aperture_sum'])
img0=img_map
print('start=',mean_delta,std_delta)

n_iter=0
while(np.abs(mean_delta)>0.0001):
    img_loop = inter_map(phot_table_center_est,new_radius,[0,0],[1,1],[480,480],method='cubic')
    #img_loop,weights_loop = inter_map_IDW(phot_table_center_est,\
    #                                      new_radius,[0,0],[1,1],[480,480],slope=1.5,\
    #                        radius_limit=0.5*5/pix_size)
    plt.imshow(np.log10(img_loop), cmap='gray_r', vmin=-3, vmax=0.15, origin='lower')
    plt.show()
    phot_table_ring_out = aperture_photometry(img_loop, aper_ring, method='subpixel',subpixels=5)
    phot_table_center_est['aperture_sum']=phot_table_in['aperture_sum']-    phot_table_ring_out['aperture_sum']
    delta=phot_table_center_in['aperture_sum']-phot_table_center_est['aperture_sum']
    mean_delta=np.nanmean(delta/phot_table_center_in['aperture_sum'])
    std_delta=np.nanstd(delta/phot_table_center_in['aperture_sum'])

    n_iter = n_iter+1
    print(n_iter,mean_delta,std_delta)
    


# In[ ]:


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.scatter(phot_table_center_in['aperture_sum'],phot_table_center_est['aperture_sum'])
ax.set_xlim([0,200])
ax.set_ylim([0,200])
ax.set_aspect('equal', adjustable='box')


# In[ ]:


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.scatter(phot_table_center_in['aperture_sum'],delta)
ax.set_xlim([0,200])
ax.set_ylim([-30,30])
print(np.nanmean(delta/phot_table_center_in['aperture_sum']),np.nanstd(delta/phot_table_center_in['aperture_sum']))


# In[ ]:





# In[ ]:





# In[ ]:




