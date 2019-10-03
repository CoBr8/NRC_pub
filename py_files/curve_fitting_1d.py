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

# define some function to play with
def gaussian(x, amp, xc, sigma):
    return amp*np.exp( -(x-xc)**2 / (2*sigma**2)) / np.sqrt(2*np.pi*sigma**2)

# set initial parameters for fit function
x = np.arange(256)
amp = 1
xc = np.median(x)
sigma = x[-1]/10
noise_factor = 0.05

# make both clean and noisy data
data = gaussian(x, amp, xc, sigma)
noise = data + noise_factor*data.max()*np.random.normal(size=data.shape)

# define some initial guess values for the fit routine
guess_vals = [amp*2, xc*0.8, sigma/1.5]

# perform the fit and calculate fit parameter errors from covariance matrix
fit_params, cov_mat = curve_fit(gaussian, x, noise, p0=guess_vals)
fit_errors = np.sqrt(np.diag(cov_mat))

# manually calculate R-squared goodness of fit
fit_residual = noise - gaussian(x, *fit_params)
fit_Rsquared = 1 - np.var(fit_residual)/np.var(noise)

print('Fit R-squared:', fit_Rsquared, '\n')
print('Fit Amplitude:', fit_params[0], '\u00b1', fit_errors[0])
print('Fit Center:   ', fit_params[1], '\u00b1', fit_errors[1])
print('Fit Sigma:    ', fit_params[2], '\u00b1', fit_errors[2])

# plotting shizzle
plt.figure(figsize=(8,4))

plt.plot(x, data, linewidth=5, color='k', label='original')
plt.plot(x, noise, linewidth=2, color='b', label='noisy')
plt.plot(x, gaussian(x, *fit_params), linewidth=3, color='r', label='fit')

plt.title('non-linear least squares fit: scipy.optimize')
plt.legend()
plt.show()

# tell LMFIT what fn you want to fit, then fit, starting iteration with guess values
lmfit_model = Model(gaussian)
lmfit_result = lmfit_model.fit(noise, x=x,
                               amp=guess_vals[0],
                               xc=guess_vals[1],
                               sigma=guess_vals[2])

# again, calculate R-squared
lmfit_Rsquared = 1 - lmfit_result.residual.var()/np.var(noise)

print('Fit R-squared:', lmfit_Rsquared, '\n')
print(lmfit_result.fit_report())

# another view of fit parameters
lmfit_result.params.pretty_print()


# plotting shizzle
plt.figure(figsize=(8,4))

plt.plot(x, data, linewidth=5, color='k', label='original')
plt.plot(x, noise, linewidth=2, color='b', label='noisy')
plt.plot(x, lmfit_result.best_fit, linewidth=3, color='r', label='fit')

plt.title('non-linear least squares fit: LMFIT package')
plt.legend()
plt.show()

# report best fit parameters with 1*sigma, 2*sigma, and 3*sigma confidence interrvals
print(lmfit_result.ci_report())

# show goodness of fits, X^2 and the reduced-X^2
print('Fit X^2:        ', lmfit_result.chisqr)
print('Fit reduced-X^2:', lmfit_result.redchi)

# access info on data set and fit performance
print('Number of Data Points:', lmfit_result.ndata)
print('Number of Fit Iterations:', lmfit_result.nfev)
print('Number of freely independent variables:', lmfit_result.nvarys)
print('Did the fit converge?:', lmfit_result.success)

# quickly check the fit residuals (input_data - fit_data)
lmfit_result.plot_residuals();
