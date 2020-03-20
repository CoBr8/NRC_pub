import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting

# Fit the data using astropy.modeling
Y, X = np.mgrid[:1024, :1024]
Z = models.Gaussian2D().evaluate(X, Y, amplitude=10, x_mean=500, y_mean=500, x_stddev=15, y_stddev=10.2, theta=18)

# amplitude=1, x_mean=0, y_mean=0, x_stddev=None, y_stddev=None, theta=None
p_init = models.Gaussian2D(x_mean=500, y_mean=500)
fit_p = fitting.LevMarLSQFitter()
p = fit_p(p_init, X, Y, Z)  # model, mesh x, mesh y, data to fit

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 18))

ax1.set_title("Data")
ax1.imshow(Z, origin='lower')

ax2.set_title("Model")
ax2.imshow(p(X, Y), origin='lower')

ax3.set_title("Residual")
ax3.imshow(Z - p(X, Y), origin='lower')

plt.show()
