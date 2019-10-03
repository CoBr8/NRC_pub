########################
# spline interpolation #
########################

import numpy as np
from scipy import interpolate
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


# BUG:

        # data = A_base
        #
        # x, y = np.mgrid[0:data.shape[0],0:data.shape[1]]
        # z    = data
        #
        # plt.figure()
        # plt.pcolor(x, y, z)
        # plt.colorbar()
        # plt.title("Sparsely sampled function.")
        # plt.show()
# :GUB

# %%

xnew, ynew = np.mgrid[-1:1:70j, -1:1:70j]
tck = interpolate.bisplrep(x, y, z, s=0)
znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)

plt.figure()
plt.pcolor(xnew, ynew, znew)
plt.colorbar()
plt.title("Interpolated function.")
plt.show()
