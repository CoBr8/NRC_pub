# First we will create a simple png generating script for all the data files in the directory
import astropy as ap
import numpy as np
import scipy as sp
import os
from astropy.io import fits
import matplotlib.pyplot as plt

# %%
dat_dir = '/home/broughtonco/documents/nrc/data/'
files   = os.listdir(dat_dir)

for file_name in files:
    if '.png' not in file_name and '.fits' in file_name:
        fits_image_filename = dat_dir+file_name
        with fits.open(fits_image_filename) as hdul:
            dat=hdul[0].data
            if len(dat.shape)==3:
                print('no')
                dat=dat[0]
                hdul.data=dat
            print(dat.shape)
            hdul.writeto(fits_image_filename[0:-5]+'_new.fits')
        plt.imshow(dat, cmap='seismic')
        plt.savefig(str(dat_dir)+'%s.png'%file_name[0:-5])
        plt.close()


# %% taking the old fits files and making them 2 dimensional

dat_dir = '/home/broughtonco/documents/nrc/data/'
files   = os.listdir(dat_dir)

for file_name in files:
    if '.png' not in file_name and '.fits' in file_name:
        fits_image_filename = dat_dir+file_name
        with fits.open(fits_image_filename) as hdul:
            dat=hdul[0].data
            if len(dat.shape)==3:
                dat=dat[0]
                hdul[0].data=dat
            hdul.writeto(fits_image_filename,overwrite=True)
        # plt.imshow(dat, cmap='seismic')
        # plt.savefig(str(dat_dir)+'%s.png'%file_name[0:-5])
        # plt.close()


# %% Generating png images with a seismic colour map.

dat_dir = '/home/broughtonco/documents/nrc/data/'
files   = os.listdir(dat_dir)

for file_name in files:
    if '.png' not in file_name and '.fits' in file_name:
        fits_FN=dat_dir+file_name
        with fits.open(fits_FN) as hdul:
            dat=hdul[0].data
            plt.imshow(dat, cmap='seismic')
            plt.savefig(str(dat_dir)+'%s.png'%file_name[0:-5])
            plt.close()


# %% Now we are going to play around with the headers files

file_name = '/home/broughtonco/documents/nrc/data/SERPM_20160202_00054_850_EA3_cal_crop_smooth_jypbm.fits'
with fits.open(file_name) as hdul:
    hdul_dat=hdul[0].data
    hdr=hdul[0].header
hdr.keys # Returns the full header
x=hdr['OBSGEO-X'] # returns the observed geocentric position on the x-axis
y=hdr['OBSGEO-Y'] # returns the observed geocentric position on the y-axis
z=hdr['OBSGEO-Z'] # returns the observed geocentric position on the z-axis
loc=dict()
loc['x']=x
loc['y']=y
loc['z']=z
OBSGEOR=(loc['x']**2+loc['y']**2+loc['z']**2)**(1/2)
hdr['OBSGEO-R']=OBSGEOR

hdr['OBSGEO-R']

hdr.keys


# %% Some Plotting of fits data

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from astropy.utils.data import download_file
from astropy.table import Table

event_filename = '/home/broughtonco/documents/nrc/data/SERPM_20160202_00054_850_EA3_cal_crop_smooth_jypbm.fits'
hdu_list = fits.open(event_filename, memmap=True)

# hdu_list[0].header
# hdu_list.info()
# evt_data = Table(hdu_list[1].data)
# evt_data

# %%
NBINS = 500
energy_hist = plt.hist(evt_data['ENERGY'], NBINS)

# %%

plot_every=100
x=evt_data['time'][::plot_every]
y=evt_data['energy'][::plot_every]

eng_scatter= plt.scatter(x, y, alpha=0.4)

# %%
plt.hexbin(x,np.array(range(len(x)))+1)

# %%

from scipy import signal
from scipy import misc
import astropy as ap
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy as sp
import os
from astropy.io import fits

fits1 = "/home/broughtonco/documents/nrc/data/NGC2071_20151226_00052_850_EA3_cal_crop_smooth_jypbm.fits"
fits2 = "/home/broughtonco/documents/nrc/data/NGC2071_20160116_00027_850_EA3_cal_crop_smooth_jypbm.fits"
with fits.open(fits1) as hdul:
    dat1=hdul[0].data
with fits.open(fits2) as hdul:
    dat2=hdul[0].data
dat1_nand=np.nan_to_num(dat1,nan=0)
dat2_nand=np.nan_to_num(dat2,nan=0)

dat_corr=signal.fftconvolve(dat1_nand,dat2_nand,mode='full')

fig,(ax_dat1,ax_dat2,ax_corr)=plt.subplots(1,3,figsize=(10,10))

ax_dat1.imshow(dat1,cmap='seismic')
ax_dat1.set_title('2015-12-26')

ax_dat2.imshow(dat2,cmap='seismic')
ax_dat2.set_title('2016-01-16')

ax_corr.imshow(dat_corr,cmap='seismic')
ax_corr.set_title('Correlated Data')


plt.show()


# %%

import astropy.io.fits as fits
inDatA='/home/broughtonco/documents/nrc/data/SERPM_20160202_00054_850_EA3_cal_crop_smooth_jypbm.fits'
inDatB='/home/broughtonco/documents/nrc/data/SerpensMain_20170724_850_IR4_ext_HK.fits'
datDiff=fits.FITSDiff(inDatA,inDatB)

print(datDiff.diff_hdus[0][1].diff_data.diff_dimensions)
