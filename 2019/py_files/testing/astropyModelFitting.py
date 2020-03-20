import numpy as np
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting

pi = np.pi
RAD2DEG = 180 / pi

# Map variable inputs
RMS1 = RMS2 = 1  # Noise level (integer repr. a percentile)
mat_size = 2048  # Size of the base map
shift = 20 * 10  # How shifted is the first map from the second?

# Map calculated inputs
side1 = side2 = mat_size * 30 // 100  # Size of the Gaussian being placed on map
gsize = (side1, side2)

noi_factor1 = RMS1 / 100  # Noise factor for first Gaussian as a multiplicative scalar
noi_factor2 = RMS2 / 100  # Noise factor for second Gaussian as a multiplicative scalar
zero_base = np.zeros((mat_size, mat_size))  # base map of zeros

x = np.arange(gsize[1])
y = np.arange(gsize[0])
xy_mesh = np.meshgrid(x, y)

# In-Gaussian variable ##
amp = 1  # amplitude

xc = np.median(x)  # centre of gaussian along x
yc = np.median(y)  # centre of gaussian along y

sigma_x = 5  # np.random.randint(5,20)               ### STD along x
sigma_y = 15  # np.random.randint(5,20)               ### STD along y

# Generating Gaussian

A1 = models.Gaussian2D().evaluate(xy_mesh[1], xy_mesh[0], amplitude=1, x_mean=xc, y_mean=yc, x_stddev=sigma_x,
                                  y_stddev=sigma_y, theta=50)  # creating gaussian dist.

sample3 = np.copy(zero_base)  # creating base map 1 from zero base
sample4 = np.copy(zero_base)  # creating base map 2 from zero base

noi1 = noi_factor1 * A1.max() * np.random.normal(size=sample3.shape)  # creating noise for map 1
noi2 = noi_factor2 * A1.max() * np.random.normal(size=sample4.shape)  # creating noise for map 2

lower = (sample3.shape[0] - side1) // 2  # centre - side to find lower bound for gaussian
upper = (sample4.shape[0] + side1) // 2  # centre + side to find upper bound for gaussian

sample3[lower:upper, lower:upper] += A1  # adding gaussian to centre of map
sample4[lower:upper, lower + shift:upper + shift] += A1  # adding gaussian shifted from centre of map

# doubling size of map to centre sample3 and sample4 with zero padding around it

base = np.zeros(shape=(sample3.shape[0] * 2, sample3.shape[1] * 2))  # generating new map

A_base = np.copy(base)  # Copying base map to new variables to edit
B_base = np.copy(base)  # Copying base map to new variables to edit

A_base[base.shape[0] // 4: 3 * base.shape[0] // 4, base.shape[1] // 4:3 * base.shape[0] // 4] += sample3 + noi1
B_base[base.shape[0] // 4: 3 * base.shape[0] // 4, base.shape[1] // 4:3 * base.shape[0] // 4] += sample4 + noi2

# calculating cross-correlation using FFT and IFFT ##

ffta = fft2(A_base)  # calculating FFT of A_base (padded map)
fftb = fft2(B_base)  # calculating FFT of B_base (padded map)

# calculating FFT based Cross-Correlation
fftc = np.roll(ifft2(ffta * fftb.conj()).real, (A_base.shape[0] - 1) // 2, axis=(0, 1))

# trimming map to be the same size as non-zero-padded inputs.
fftc = fftc[fftc.shape[0] // 4:3 * fftc.shape[0] // 4, fftc.shape[1] // 4:3 * fftc.shape[0] // 4]

A_base = A_base[A_base.shape[0] // 4: 3 * A_base.shape[0] // 4, A_base.shape[0] // 4: 3 * A_base.shape[0] // 4]
B_base = B_base[B_base.shape[0] // 4: 3 * B_base.shape[0] // 4, B_base.shape[0] // 4: 3 * B_base.shape[0] // 4]

side = 200  # must be even for now ## a simple if/else would allow it to be even/odd trivial to code.
x, y = np.arange(0, side, 1), np.arange(0, side, 1)
# noinspection PyTypeChecker
fit_mesh: np.ndarray = np.meshgrid(x, y)
fit_xc = np.median(x)
fit_yc = np.median(y)

y_loc, x_loc = np.where(fftc == fftc.max())
x_loc, y_loc = x_loc[0], y_loc[0]
fit_fftc = fftc[y_loc - side // 2:y_loc + side // 2, x_loc - side // 2:x_loc + side // 2]

gauss_init = models.Gaussian2D(x_mean=fit_fftc.shape[0] // 2, y_mean=fit_fftc.shape[0] // 2)
fitting_gauss = fitting.LevMarLSQFitter()
best_fit_gauss = fitting_gauss(gauss_init, fit_mesh[1], fit_mesh[0], fit_fftc)
cov_diag = np.diag(fitting_gauss.fit_info['param_cov'])

mod = best_fit_gauss(fit_mesh[1], fit_mesh[0])
Residuals = fit_fftc - mod
RSquared = 1 - np.var(fit_fftc - mod) / np.var(mod)

if RSquared < -1 or RSquared > 1:
    raise Exception('No fit')

amp_per = np.sqrt(cov_diag[0]) / best_fit_gauss.amplitude.value * 100
mean_x_per = np.sqrt(cov_diag[1]) / best_fit_gauss.x_mean.value * 100
mean_y_per = np.sqrt(cov_diag[2]) / best_fit_gauss.y_mean.value * 100
std_x_per = np.sqrt(cov_diag[3]) / best_fit_gauss.x_stddev.value * 100
std_y_per = np.sqrt(cov_diag[4]) / best_fit_gauss.y_stddev.value * 100
theta_per = np.sqrt(cov_diag[5]) / best_fit_gauss.theta.value * 100

amp_str = 'Amplitude: {:5g} \u00b1 {:5g} ({:.3g}%)'.format(
    best_fit_gauss.amplitude.value,
    np.sqrt(cov_diag[0]),
    amp_per)
mean_x_str = 'X Mean:    {:5g} \u00b1 {:5g} ({:.3g}%)'.format(
    best_fit_gauss.x_mean.value,
    np.sqrt(cov_diag[1]),
    mean_x_per)
mean_y_str = 'Y Mean:    {:5g} \u00b1 {:5g} ({:.3g}%)'.format(
    best_fit_gauss.y_mean.value,
    np.sqrt(cov_diag[2]),
    mean_y_per)
std_x_str = 'X STD:     {:5g} \u00b1 {:5g} ({:.3g}%)'.format(
    best_fit_gauss.x_stddev.value,
    np.sqrt(cov_diag[3]),
    std_x_per)
std_y_str = 'Y STD:     {:5g} \u00b1 {:5g} ({:.3g}%)'.format(
    best_fit_gauss.y_stddev.value,
    np.sqrt(cov_diag[4]),
    std_y_per)
theta_str = '???Theta:  {:5g} \u00b1 {:5g} ({:.3g}%)'.format(
    best_fit_gauss.theta.value * RAD2DEG % 90,
    np.sqrt(cov_diag[5]) * RAD2DEG % 90,
    abs(theta_per))

plt.close()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 12))

ax1.set_title("Data")
ax1.imshow(fit_fftc, origin='lower')
ax1.set_xlim(left=fit_fftc.shape[1], right=0)
ax1.set_ylim(bottom=0, top=fit_fftc.shape[0])

ax2.set_title("Model")
ax2.imshow(mod, origin='lower')
ax2.set_xlim(left=mod.shape[1], right=0)
ax2.set_ylim(bottom=0, top=mod.shape[0])

ax3.set_title("Residual")
ax3.imshow(Residuals, origin='lower')
ax3.set_xlim(left=Residuals.shape[1], right=0)
ax3.set_ylim(bottom=0, top=Residuals.shape[0])

fig.text(0.3, 0.15, 'R-Squared: {}'.format(RSquared))
fig.text(0.3, 0.05, '{}\n{}\n{}\n{}\n{}\n{}'.format(
    amp_str,
    mean_x_str,
    mean_y_str,
    std_x_str,
    std_y_str,
    theta_str))
# fig.savefig('/home/broughtonco/documents/nrc_pub/AstropyFitting200x200_5.png')
plt.show()
