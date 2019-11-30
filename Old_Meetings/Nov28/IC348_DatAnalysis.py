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

MetaData = np.loadtxt('/home/broughtonco/documents/nrc/data/IC348/IC348_850_EA3_cal_metadata.txt', dtype='str')

BaseDate = MetaData[0][2][1:]
Date = MetaData[Epoch - 1][2][1:]
r = radius
r2 = radius ** 2

hdr = 'Epoch JD Elevation T225 RMS Cal_f Cal_f_err ' \
      'B_5 B_Err_5 m_5 m_5_err ' \
      'B_10 B_Err_10 m_10 m_10_err ' \
      'B_15 B_Err_15 m_15 m_15_err'
li = np.zeros(19)
Dat_dict = {}

for dist in [5, 10, 15]:
    num = len(radius[radius <= dist])

    DivACData, DivPSDData = [], []
    fit_ac, cov_ac = [], []
    err_ac = []

    for i in range(0, len(AC_Data)):
        DivAC = AC_Data[i] / AC_Data[0]
        DivACData.append(DivAC)
        optimal_fit_AC, cov_mat_AC = curve_fit(f, r2[1:num], DivAC[1:num])
        fit_ac.append(optimal_fit_AC)
        cov_ac.append(cov_mat_AC)
        err_ac.append(np.sqrt(np.diag(cov_mat_AC)))
    Dat_dict[dist] = [fit_ac, err_ac]
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

    AC_DIV.set_xlabel('$r^2$ from centre')
    AC_DIV.set_ylabel('Similarity in AC')

    plt.suptitle('Epoch {} / Epoch {}'.format(Date, BaseDate))
    fig.savefig('/home/broughtonco/documents/nrc/data/IC348/IC348_plots/IC348_linearFit_radius_{}.png'.format(dist))
    # plt.show()

for epoch in range(len(AC_Data)):
    e_num = MetaData[epoch][0]
    jd = MetaData[epoch][4]
    elev = MetaData[epoch][6]
    t225 = MetaData[epoch][7]
    rms = MetaData[epoch][8]
    cal_f = MetaData[epoch][10]
    cal_f_err = MetaData[epoch][11]

    # dictionary[key][fit (0) or err (1)][epoch number][m (0) or b (1)]
    m5 = Dat_dict[5][0][epoch][0]
    m5_err = Dat_dict[5][1][epoch][0]

    m10 = Dat_dict[10][0][epoch][0]
    m10_err = Dat_dict[10][1][epoch][0]

    m15 = Dat_dict[15][0][epoch][0]
    m15_err = Dat_dict[15][1][epoch][0]

    b5 = Dat_dict[5][0][epoch][1]
    b5_err = Dat_dict[5][1][epoch][1]

    b10 = Dat_dict[10][0][epoch][1]
    b10_err = Dat_dict[10][1][epoch][1]

    b15 = Dat_dict[15][0][epoch][1]
    b15_err = Dat_dict[15][1][epoch][1]

    dat = np.array([e_num, jd, elev, t225, rms, cal_f, cal_f_err,
                    b5, b5_err, m5, m5_err,
                    b10, b10_err, m10, m10_err,
                    b15, b15_err, m15, m15_err],
                   dtype='float'
                   )
    li = np.vstack((li, dat))

frmt = '% 3d % 14f % 4d % 5f % 8f % 8f % 8f % 8f % 8f % 8f % 8f % 8f % 7f % 9f % 8f % 8f % 8f % 9f % 8f'
form = '%s'

np.savetxt('/home/broughtonco/documents/nrc/data/IC348/Datafiles/IC348_TABLE_TOPCAT.table',
           li[1:],
           fmt=form,
           header=hdr
           )
np.savetxt('/home/broughtonco/documents/nrc/data/IC348/Datafiles/IC348_TABLE_READABLE.table',
           li[1:],
           fmt=frmt,
           header=hdr
           )
