import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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


Epoch = 5
row, col = 4, 4

AC_Data = np.load('/home/broughtonco/documents/nrc/data/IC348/Datafiles/IC348_AC.npy')
PSD_Data = np.load('/home/broughtonco/documents/nrc/data/IC348/Datafiles/IC348_PSD.npy')
radius = np.load('/home/broughtonco/documents/nrc/data/IC348/Datafiles/IC348_radius.npy')

r2 = radius ** 2

MetaData = np.loadtxt('/home/broughtonco/documents/nrc/data/IC348/IC348_850_EA3_cal_metadata.txt', dtype='str')

BaseDate = MetaData[0][2][1:]
Date = MetaData[Epoch - 1][2][1:]

for dist in [5, 10, 15, 20]:
    num = len(radius[radius <= dist])

    DivACData = []
    fit_ac, cov_ac = [], []

    for i in range(0, len(AC_Data)):
        DivAC = AC_Data[i] / AC_Data[0]
        DivACData.append(DivAC)
        optimal_fit_AC, cov_mat_AC = curve_fit(f, r2[1:num], DivAC[1:num])
        fit_ac.append(optimal_fit_AC)
        cov_ac.append(cov_mat_AC)

    # Plotting follows
    AC_text = 'y = {:g} \u00B1 {:g} x + {:g} \u00B1 {:g}'.format(fit_ac[Epoch - 1][0],
                                                                 np.sqrt(cov_ac[Epoch - 1][0, 0]),
                                                                 fit_ac[Epoch - 1][1],
                                                                 np.sqrt(cov_ac[Epoch - 1][1, 1])
                                                                 )

    fig = plt.figure(figsize=(10, 10))
    grid = plt.GridSpec(nrows=row, ncols=col, hspace=0.5, wspace=0.2)

    fig.text(0.5, 13 / (row*col), AC_text, horizontalalignment='center', verticalalignment='center')

    AC_DIV = fig.add_subplot(grid[:, :])
    AC_DIV.set_title('AC Div Map')
    AC_DIV.scatter(r2[1:num], DivACData[Epoch - 1][1:num].real, s=1, marker=',', lw=0, color='red')
    AC_DIV.plot(r2[1:num], f(r2[1:num], fit_ac[Epoch - 1][0], fit_ac[Epoch - 1][1]),
                linestyle='--',
                color='green',
                linewidth=0.5)
    # AC_DIV.set_xticks(np.arange(0, dist+1, step=1))
    AC_DIV.set_xlabel('radius from centre')
    AC_DIV.set_ylabel('Similarity in AC')

    plt.suptitle('Epoch {} / Epoch {}'.format(Date, BaseDate))
    # fig.savefig('/home/broughtonco/documents/nrc/data/IC348/IC348_plots/IC348_linearFit_radius_{}.png'.format(dist))
    plt.show()
