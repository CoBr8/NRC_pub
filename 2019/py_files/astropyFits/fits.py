#################################################
# Basics of fits file creation and manipulation #
#################################################

from astropy.io import fits
import numpy as np
import datetime as dt

tdy=dt.date.today()
fmt='%Y-%m-%d'
tdy=tdy.strftime(fmt)

# %%
x=np.arange(100.0)
y=np.arange(100.0)
Z=np.meshgrid(x,y)
Z=Z[0]+Z[1]

# %%
import os
os.remove('/home/broughtonco/documents/nrc/py_files/new2.fits')
hdu=fits.PrimaryHDU(Z)
hdu.writeto('/home/broughtonco/documents/nrc/py_files/new2.fits')

# %%
tdy=dt.date.today()
fmt='%Y-%m-%d'
tdy=tdy.strftime(fmt)
cmt1='I created this file to play with astropy fits creation'
hst1='File created on %s' %tdy
with fits.open('/home/broughtonco/documents/nrc/py_files/new2.fits') as hdu:
    hduInfo=hdu.info()
    hduData=hdu[0].data
    hduHead=hdu[0].header
    hduHead.set('Date',tdy)
    hduHead.set('Comment',cmt1)
    hduHead.set('History',hst1)
    hduHead.set('Observer','Colton J Broughton')
    hduHead.set('History','I edited this file at 10:30')
    hduHead.set('comment','I tested with this file at 10:31')

list(hduHead.keys())


# %%

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from astropy.io import fits
import datetime as dt
import os

def makeGaussian(size, fwhm = 3,amp = 1, centre=None):
    """
    Make a square gaussian kernel.
    ==============================
    IN:
    ---
        size:   is the length of a side of the square
        fwhm:   is full-width-half-maximum, which can
                be thought of as an effective radius.
        amp:    amplitude of the gaussian, default 1
        centre: the center of the gaussian, default (0,0)

    OUT:
    ----
        square gaussian kernel

    END
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if centre is None:
        x0 = y0 = size // 2
    else:
        x0 = centre[0]
        y0 = centre[1]

    return amp*np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

end=20
shift=20; RMS = 10
fwhm_A=4
fwhm_B=1
B = np.zeros( ( 100, 100 ) )
noi1 = np.random.randn(*B.shape)*(RMS/100)
noi2 = np.random.randn(*B.shape)*(RMS/100)
A1 = makeGaussian(40, fwhm=fwhm_A, centre=None)
A2 = makeGaussian(40, fwhm=fwhm_B, centre=None)

nb = B.shape[0];na = A1.shape[0]
lower = (nb // 2) - (na // 2); upper = (nb // 2) + (na // 2)
sample3 = np.copy(B); sample4 = np.copy(B)
sample3[ lower:upper, lower:upper ] += A1
sample4[ lower:upper, lower+shift:upper+shift ] += A2
sample3 += noi1; sample4 += noi2
corr = signal.correlate2d(sample3, sample4, boundary='fill', mode='same')
corr[corr.shape[0]//2][corr.shape[1]//2]=np.random.randn()*(RMS/100)
os.remove('/home/broughtonco/documents/nrc/py_files/gaussian.fits')
hdu=fits.PrimaryHDU([sample3,sample4,corr])
hdu.writeto('/home/broughtonco/documents/nrc/py_files/gaussian.fits')
FWHM_B=[]
for i in range(1,end+1):
    fwhm_B=i
    FWHM_B.append(i)
    sample4 = np.copy(B)
    A2 = makeGaussian(40, fwhm=fwhm_B, centre=None)
    sample4[ lower:upper, lower+shift:upper+shift ] += A2
    sample4 += noi2;
    corr = signal.correlate2d(sample3, sample4, boundary='fill', mode='same')
    dat=np.array([sample3,sample4,corr])
    fits.append( '/home/broughtonco/documents/nrc/py_files/gaussian.fits', dat )

# %%

tdy=dt.date.today()
fmt='%Y-%m-%d'
tdy=tdy.strftime(fmt)

cmt1='I created this file to play with astropy fits'
hst1='File created on %s' %tdy

for i in range(end):
    with fits.open('/home/broughtonco/documents/nrc/py_files/gaussian.fits') as hdu:
        hduInfo=hdu.info()
        hduData=hdu[i].data
        hduHead=hdu[i].header
        hduHead.set('Date',tdy)
        hduHead.set('Comment',cmt1)
        hduHead.set('History',hst1)
        hduHead.set('Observer','Colton J Broughton')

    fig, ((ax11,ax12,ax13),(ax21,ax22,ax23)) = plt.subplots(2,3,sharex=True,figsize=(10,10))

    ax11.imshow(hduData[0],cmap='seismic')
    ax11.axhline(y=sample3.shape[1]//2)
    ax11.set_title('Source 1')

    ax12.imshow(hduData[1],cmap='seismic')
    ax12.axhline(y=sample4.shape[1]//2)
    ax12.set_title('Source 2')

    ax13.imshow(hduData[2],cmap='seismic')
    ax13.axhline(y=corr.shape[1]//2)
    ax13.set_title('Correlated')

    ax21.plot(hduData[0][:][sample3.shape[1]//2])
    ax22.plot(hduData[1][:][sample3.shape[1]//2])
    ax23.plot(hduData[2][:][sample3.shape[1]//2])

    plt.figtext(0.15, 0.58, 'fwhm=%f'%(fwhm_A), horizontalalignment='left', wrap=True)
    plt.figtext(0.48, 0.58, 'fwhm=%f'%(FWHM_B[i]), horizontalalignment='left', wrap=True)

    fig.tight_layout()
    fig.savefig('/home/broughtonco/documents/nrc/data/images/correlation/GC_sliced_%s.png' %str(i+1))
    fig.show()


# %%
