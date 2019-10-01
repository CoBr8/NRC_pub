import numpy as np
from math import *
from scipy.optimize import curve_fit
from lmfit import Model
from astropy.io import fits
import os

%matplotlib inline
import matplotlib.pyplot as plt

def gaussian_2d(xy_mesh, amp, xc, yc, sigma_x, sigma_y):

    # unpack 1D list into 2D x and y coords
    (x, y) = xy_mesh

    # make the 2D Gaussian matrix
    gauss = amp*np.exp(-((x-xc)**2/(2*sigma_x**2)+(y-yc)**2/(2*sigma_y**2)))/(2*np.pi*sigma_x*sigma_y)

    # flatten the 2D Gaussian down to 1D
    return np.ravel(gauss)

with fits.open('/home/broughtonco/documents/nrc/py_files/gc_five_source.fits') as hdu:
    # hdu.info()
    data=hdu[0].data
    hdr=hdu[0].header

# data.shape ## first is ideal, second is noisey, 3rd is correlation

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,6))
ax1.imshow(data[1],origin='bottom')
# data[1].mean()**(1/2)


# loc=np.where(data[1]==data[1].mean()**(2/5))
loc=np.where(data[1]==data[1].max())
ax1.scatter( loc[1], loc[0], color='r' )
x = int(loc[1]-5)
y = int(loc[0]-5)
GC=data[1][y:y+10,x:x+10]
ax2.imshow(GC, origin='bottom')
fig.savefig('/home/broughtonco/documents/nrc/data/images/fitting/10x10Fitting.png')
