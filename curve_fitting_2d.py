## source: https://kippvs.com/2018/06/non-linear-fitting-with-python/ ##


# package imports
from math import *
import numpy as np
from scipy.optimize import curve_fit
from lmfit import Model

%matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = '#e8e8e8'
plt.rcParams['axes.edgecolor'] = '#e8e8e8'
plt.rcParams['figure.facecolor'] = '#e8e8e8'

def gaussian_2d(xy_mesh, amp, xc, yc, sigma_x, sigma_y):

    # unpack 1D list into 2D x and y coords
    (x, y) = xy_mesh

    # make the 2D Gaussian matrix
    gauss = amp*np.exp(-((x-xc)**2/(2*sigma_x**2)+(y-yc)**2/(2*sigma_y**2)))/(2*np.pi*sigma_x*sigma_y)

    # flatten the 2D Gaussian down to 1D
    return np.ravel(gauss)


###############
## change me ##
###############
n = 2500
noise_factor = 0.02
###############
## change me ##
###############

    # create the 1D list (xy_mesh) of 2D arrays of (x,y) coords
x = np.arange(n)
y = np.arange(n)
xy_mesh = np.meshgrid(x,y)

# set initial parameters to build mock data
amp = 1
xc, yc = np.median(x), np.median(y)
sigma_x, sigma_y = x[-1]/10, y[-1]/10


# make both clean and noisy data, reshaping the Gaussian to proper 2D dimensions
data = gaussian_2d(xy_mesh, amp, xc, yc, sigma_x, sigma_y).reshape(np.outer(x, y).shape)
noise = data + noise_factor*data.max()*np.random.normal(size=data.shape)

# plot the function and with noise added
fig, (ax1,ax2) =plt.subplots(1,2,figsize=(12,6))

ax1.set_title('model')
ax1.imshow(data, origin='bottom')
ax1.grid(visible=False)

ax2.set_title('noisy data')
ax2.imshow(noise, origin='bottom')
ax2.grid(visible=False)

fig.text(0.48,0,'n = %d'%n)

plt.show()

# define some initial guess values for the fit routine
guess_vals = [amp*2, xc*0.8, yc*0.8, sigma_x/1.5, sigma_y/1.5]

# perform the fit, making sure to flatten the noisy data for the fit routine
fit_params, cov_mat = curve_fit(gaussian_2d, xy_mesh, np.ravel(noise), p0=guess_vals)

# calculate fit parameter errors from covariance matrix
fit_errors = np.sqrt(np.diag(cov_mat))

# manually calculate R-squared goodness of fit
fit_residual = noise - gaussian_2d(xy_mesh, *fit_params).reshape(np.outer(x,y).shape)
fit_Rsquared = 1 - np.var(fit_residual)/np.var(noise)


print('\nFit R-squared:', fit_Rsquared, '\n')

print('Fit Amplitude:', fit_params[0], '\u00b1', \
                        fit_errors[0], '\n%.4g%%' \
                        %(fit_errors[0]/fit_params[0]*100))
print('Fit X-Center: ', fit_params[1], '\u00b1', \
                        fit_errors[1],'\n%.4g%%' \
                        %(fit_errors[1]/fit_params[1]*100))
print('Fit Y-Center: ', fit_params[2], '\u00b1', \
                        fit_errors[2],'\n%.4g%%' \
                        %(fit_errors[2]/fit_params[2]*100))
print('Fit X-Sigma:  ', fit_params[3], '\u00b1', \
                        fit_errors[3],'\n%.4g%%' \
                        %(fit_errors[3]/fit_params[3]*100))
print('Fit Y-Sigma:  ', fit_params[4], '\u00b1', \
                        fit_errors[4],'\n%.4g%%' \
                        %(fit_errors[4]/fit_params[4]*100))

with open('/home/broughtonco/documents/nrc/data/fitting/fittment.txt', 'w') as f:
    f.write('Fit R-squared: {:f}\n'.format(fit_Rsquared))
    f.write("Fit Amplitude: {:f} {:s} {:f} ({:.4g}%)\n".format(\
                fit_params[0], '\u00b1', fit_errors[0], fit_errors[0]/fit_params[0]*100))
    f.write('Fit X-Center:  {:f} {:s} {:f} ({:.4g}%)\n'.format(\
                fit_params[1], '\u00b1', fit_errors[1], fit_errors[1]/fit_params[1]*100))
    f.write('Fit Y-Center:  {:f} {:s} {:f} ({:.4g}%)\n'.format(\
                fit_params[2], '\u00b1', fit_errors[2], fit_errors[2]/fit_params[2]*100))
    f.write('Fit X-Sigma:   {:f} {:s} {:f} ({:.4g}%)\n'.format(\
                fit_params[3], '\u00b1', fit_errors[3], fit_errors[3]/fit_params[3]*100))
    f.write('Fit Y-Sigma:   {:f} {:s} {:f} ({:.4g}%)\n'.format(\
                fit_params[4], '\u00b1', fit_errors[4], fit_errors[4]/fit_params[4]*100))


# check against actual parameter values
amp, xc, yc, sigma_x, sigma_y

# set contour levels out to 3 sigma
sigma_x_pts = xc + [sigma_x, 2*sigma_x, 3*sigma_x]
sigma_y_pts = yc + [sigma_y, 2*sigma_y, 3*sigma_y]
sigma_xy_mesh = np.meshgrid(sigma_x_pts, sigma_y_pts)

contour_levels = gaussian_2d(sigma_xy_mesh, amp, xc, yc,
                             sigma_x, sigma_y).reshape(sigma_xy_mesh[0].shape)
contour_levels = list(np.diag(contour_levels)[::-1])

# make labels for each contour
labels = {}
label_txt = [r'$3\sigma$', r'$2\sigma$', r'$1\sigma$']
for level, label in zip(contour_levels, label_txt):
    labels[level] = label

# plot the function with noise added

plt.title('probability coverage')
plt.imshow(noise, origin='lower')
CS = plt.contour(data, levels=contour_levels, colors=['red', 'orange', 'white'])
plt.clabel(CS, fontsize=16, inline=1, fmt=labels)
plt.grid(visible=False)
plt.show()

# %%

# create a zoomed view of the noisy data, using fit error for scaling
x_zoom = np.linspace(xc-7*fit_errors[1], xc+7*fit_errors[1], n)
y_zoom = np.linspace(xc-5*fit_errors[2], xc+5*fit_errors[2], n)
xy_mesh_zoom = np.meshgrid(x_zoom,y_zoom)


# make noisy data using same parameters as before, except zoomed to center
data_zoom = gaussian_2d(xy_mesh_zoom, amp, xc, yc,
                        sigma_x, sigma_y).reshape(np.outer(x_zoom,y_zoom).shape)
noise_zoom = data_zoom + noise_factor*data_zoom.max()*np.random.normal(size=data_zoom.shape)

# set contour levels out to 3 standard deviations
err_x_pts = xc + [fit_errors[1], 2*fit_errors[1], 3*fit_errors[1]]
err_y_pts = yc + [fit_errors[2], 2*fit_errors[2], 3*fit_errors[2]]
err_xy_mesh = np.meshgrid(err_x_pts, err_y_pts)

extent = [x_zoom[0], x_zoom[-1], y_zoom[0], y_zoom[-1]]

contour_levels = gaussian_2d(err_xy_mesh, amp, xc, yc,
                             sigma_x, sigma_y).reshape(err_xy_mesh[0].shape)
contour_levels = list(np.diag(contour_levels)[::-1])

# make labels for each contour
labels = {}
label_txt = [r'$3\sigma$', r'$2\sigma$', r'$1\sigma$']
for level, label in zip(contour_levels, label_txt):
    labels[level] = label

# plot the function with noise added
plt.figure(figsize=(6,6))
plt.title('fit uncertainty for $(x_c, y_c)$')
plt.imshow(noise_zoom, origin='lower', extent=extent)
CS = plt.contour(data_zoom, levels=contour_levels, origin='lower',
                 colors=['red', 'orange', 'white'], extent=extent)
plt.clabel(CS, fontsize=16, inline=1, fmt=labels)
plt.grid(visible=False)
plt.show()

# tell LMFIT what fn you want to fit, then fit, starting iteration with guess values
lmfit_model = Model(gaussian_2d)
lmfit_result = lmfit_model.fit(np.ravel(noise),
                               xy_mesh=xy_mesh,
                               amp=guess_vals[0],
                               xc=guess_vals[1],
                               yc=guess_vals[2],
                               sigma_x=guess_vals[3],
                               sigma_y=guess_vals[4])

# again, calculate R-squared
lmfit_Rsquared = 1 - lmfit_result.residual.var()/np.var(noise)

print('Fit R-squared:', lmfit_Rsquared, '\n')
print(lmfit_result.fit_report())

lmfit_result.params.pretty_print()

# check against actual parameter values
amp, xc, yc, sigma_x, sigma_y
