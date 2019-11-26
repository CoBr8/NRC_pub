import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.modeling.models import Gaussian2D
import operator as op
from astropy.io import fits


# noinspection PyShadowingNames
def colourbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    figle = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return figle.colorbar(mappable, cax=cax, format='%g')


def f(x, m, b):
    y = m * x + b
    return y


def auto_correlate(epoch=None, clipped_side=400):
    from numpy.fft import fft2, ifft2, fftshift
    if epoch is None:
        raise Exception('You need to pass a 2D map for this function to work')
    else:
        mid_map_x, mid_map_y = epoch.shape[1] // 2, epoch.shape[0] // 2
        clipped_epoch = epoch[mid_map_y - clipped_side // 2:mid_map_y + clipped_side // 2,
                              mid_map_x - clipped_side // 2:mid_map_x + clipped_side // 2
                              ]
    ac = ifft2(fft2(clipped_epoch)*fft2(clipped_epoch).conj())
    return fftshift(ac)


X, Y = np.meshgrid(np.arange(0, 800), np.arange(0, 800))

amp_1 = 2
FWHM_1 = 20
sigma_1 = FWHM_1 / np.sqrt(8*np.log(2))

amp_2 = 1
FWHM_2 = 15
sigma_2 = FWHM_2 / np.sqrt(8*np.log(2))

midx = X.shape[1]//2
midy = X.shape[0]//2

# amp, x_mean, y_mean, x_sigma, y_sigma, theta
p1 = [amp_1, midx, midy, sigma_1, sigma_1, 0]
p2 = [amp_2, midx, midy, sigma_2, sigma_2, 0]

gauss = Gaussian2D()

map_1 = gauss.evaluate(X, Y, *p1)
map_2 = gauss.evaluate(X, Y, *p2)

ac_map_1 = auto_correlate(map_1)
ac_map_2 = auto_correlate(map_2)

ac_division_map = ac_map_1 / ac_map_2
# +====================================================================================================================+
'''
The following code is for sorting the data based on the radius.
We then try to fit a line to determine the slope and intercept values for this data.
'''
# lists to append the 1D data
ac_div_data = []
radius = []

length = ac_division_map.shape[0]  # the map is square always so we're not worried about it.
midx = ac_division_map.shape[1]//2
midy = ac_division_map.shape[0]//2

# determining all of the locations in the map using a cartesian product function
loc = list(
    product(
        np.arange(0, length),
        np.arange(0, length)
    )
)
for i, j in loc:
    r = ((i - midx) ** 2 + (j - midy) ** 2) ** (1 / 2)
    radius.append(r)
    ac_div_data = np.append(ac_div_data, ac_division_map[j, i])

# Sorting the data based on the radius
r, ac_div_data = zip(
    *sorted(
        list(zip(radius, ac_div_data)),
        key=op.itemgetter(0)
    )
)

# ensuring they remain numpy arrays, easier to manipulate!
r = np.array(r)
ac_div_data = np.array(ac_div_data)

# determining how many data points are in the first 10 pixel radius
num = len(r[r <= 20])

# squaring all the points in r so as to find a "more" linear fit.
r2 = r**2

# fitting a curve using r^2 and our comparison map
fit_AC, cov_mat_AC = curve_fit(f, r2[1:num], ac_div_data[1:num])
AC_text = 'y = {:g} \u00B1 {:g} x + {:g} \u00B1 {:g}'.format(fit_AC[0],
                                                             np.sqrt(cov_mat_AC[0, 0]),
                                                             fit_AC[1],
                                                             np.sqrt(cov_mat_AC[1, 1])
                                                             )

fig = plt.figure(figsize=(10, 10))
grid = plt.GridSpec(nrows=1, ncols=1, hspace=0.5, wspace=0.2)

AC_DIV = fig.add_subplot(grid[:, :])

AC_DIV.set_title('AC Div Map')
AC_DIV.scatter(r2[1:num], ac_div_data[1:num].real, s=1, marker=',', lw=0, color='red')
AC_DIV.plot(r2[1:num],
            f(r2[1:num], fit_AC[0], fit_AC[1]),
            linestyle='--',
            color='green',
            linewidth=0.5
            )

AC_DIV.set_xlabel('$r^2$ from centre')
AC_DIV.set_ylabel('Similarity in AC')

fig.text(0.5, 0.5, AC_text, horizontalalignment='center', verticalalignment='center')

plt.suptitle('{} / {}'.format('G1', "G2"))
# plt.savefig('/home/broughtonco/documents/nrc_pub/Nov26/ac_div_map.png')
plt.show()