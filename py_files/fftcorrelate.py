###########
# imports #
###########
import astropy.io.fits as fits
import numpy as np
from numpy import conj as conjugate
from numpy.fft import fft2,ifft2
from scipy import signal
%matplotlib inline
import matplotlib.pyplot as plt
#############
# functions #
#############

def gaussian_2d(xy_mesh, amp, xc, yc, sigma_x, sigma_y):

    # unpack 1D list into 2D x and y coords
    (x, y) = xy_mesh

    # make the 2D Gaussian matrix
    gauss = amp*np.exp(-((x-xc)**2/(2*sigma_x**2)+(y-yc)**2/(2*sigma_y**2)))/(2*np.pi*sigma_x*sigma_y)

    # flatten the 2D Gaussian down to 1D
    return gauss

################
## Parameters ##
################
def main():
### Parameters ##
    shift = -20*10;

    RMS1 = RMS2 = 2

    percent_factor = RMS2/100

    mat_size = 2000
    side = 200

    B     = np.zeros( ( mat_size, mat_size ) )
    gsize = (side, side)

### create the 1D list (xy_mesh) of 2D arrays of (x,y) coords
    x,y = np.arange(gsize[1]), np.arange(gsize[0])
    xy_mesh = np.meshgrid(x,y)
    amp = 1
    xc,yc = np.median(x),np.median(y)
    sigma_x, sigma_y = x[-1]/10, y[-1]/6
    A1 = gaussian_2d(xy_mesh,amp,xc,yc,sigma_x,sigma_y)

###
    sample3 = np.copy(B); sample4 = np.copy(B)
    noi1    = percent_factor * A1.max() * np.random.normal(size=B.shape)
    noi2    = percent_factor * A1.max() * np.random.normal(size=B.shape)

    X=np.random.randint(0,1024-side)
    Y=np.random.randint(0,1024-side)
    lower=(B.shape[0]-side)//2
    upper=(B.shape[0]+side)//2
    sample3[ lower:upper, lower:upper ] += A1
    sample4[ lower:upper, lower+shift:upper+shift] += A1

    A = sample3
    B = sample4

    base = np.zeros(shape=(A.shape[0]*2,A.shape[1]*2))

    A_base = np.copy(base)
    B_base = np.copy(base)

    A_base[A.shape[0]//2:3*A.shape[0]//2, A.shape[1]//2:3*A.shape[0]//2] += A + np.sqrt(noi1**2)
    B_base[B.shape[0]//2:3*B.shape[0]//2, B.shape[1]//2:3*B.shape[0]//2] += B + np.sqrt(noi2**2)

    data_array=[ [A_base,A_base],[A_base,B_base],\
                 [B_base,A_base],[B_base,B_base] ]

    name_array=[ 'corr(A,A)', 'corr(A,B)', 'corr(B,A)', 'corr(B,B)' ]

    for i in range(len(data_array)):
        ffta=fft2(data_array[i][0])

        fc = np.roll( ifft2( fft2(data_array[i][1]) * ffta.conj() ).real, (A_base.shape[0] - 1) // 2, axis = (0,1) )

        fig1,((ax1,ax2,ax3),(axslice1,axslice2,axslice3))=plt.subplots(2,3,figsize=(18,12),sharex=True,sharey='row')
        # fig2,(axslice1,axslice2,axslice3)=plt.subplots(1,3,figsize=(18,6),sharex=True,sharey=True)

        ax1.imshow(data_array[i][0], origin='bottom')
        ax1.set_title(name_array[i][-4])

        ax2.imshow(data_array[i][1], origin='bottom')
        ax2.set_title(name_array[i][-2])

        ax3.imshow(fc,origin='bottom')
        ax3.set_title(name_array[i])

        axslice1.plot(data_array[i][0][data_array[i][0].shape[0]//2])
        axslice2.plot(data_array[i][1][data_array[i][1].shape[0]//2])
        axslice3.plot(fc[fc.shape[0]//2])

        fig1.text(0.5,0.5,name_array[i]+'\n'+'| Noise |')
        plt.savefig('/home/broughtonco/documents/nrc/data/images/fft_correlation/fft_correlation_{:s}.png'\
                                                                    .format(name_array[i])\
                                                                    )
        plt.show()

import timeit
timeit.timeit(stmt='main()',number=10000)
