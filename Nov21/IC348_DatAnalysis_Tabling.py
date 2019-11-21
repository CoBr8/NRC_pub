import numpy as np
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


AC_Data = np.load('/home/broughtonco/documents/nrc/data/IC348/Datafiles/IC348_AC.npy')
PSD_Data = np.load('/home/broughtonco/documents/nrc/data/IC348/Datafiles/IC348_PSD.npy')
radius = np.load('/home/broughtonco/documents/nrc/data/IC348/Datafiles/IC348_radius.npy')

MetaData = np.loadtxt('/home/broughtonco/documents/nrc/data/IC348/IC348_850_EA3_cal_metadata.txt', dtype='str')

BaseDate = MetaData[0][2][1:]
hdr = 'Epoch, Julian Date, Elevation, Tau-225, RMS, Cal_f , Cal_f err, ' \
      'B (5), B Err (5), m / B (5), ' \
      'B (10), B Err (10), m / B (10), ' \
      'B (15), B Err (15), m / B (15) '
li = np.zeros(16)
Dat_dict = {}
for dist in [5, 10, 15]:
    num = len(radius[radius <= dist])

    DivACData, DivPSDData = [], []
    fit_ac, cov_ac = [], []
    fit_psd, cov_psd = [], []
    err_ac, err_psd = [], []

    for i in range(0, len(AC_Data)):
        DivAC = AC_Data[i] / AC_Data[0]
        DivACData.append(DivAC)
        optimal_fit_AC, cov_mat_AC = curve_fit(f, radius[1:num], DivAC[1:num])
        fit_ac.append(optimal_fit_AC)
        cov_ac.append(np.diag(cov_mat_AC))
        err_ac.append(np.sqrt(np.diag(cov_mat_AC)))

        DivPSD = PSD_Data[i] / PSD_Data[0]
        DivPSDData.append(DivPSD)
        optimal_fit_PSD, cov_mat_PSD = curve_fit(f, radius[1:num], DivPSD[1:num])
        fit_psd.append(optimal_fit_PSD)
        cov_psd.append(np.diag(cov_mat_PSD))
        err_psd.append(np.sqrt(np.diag(cov_mat_PSD)))
    Dat_dict[dist] = [fit_ac, err_ac]

for epoch in range(len(AC_Data)):
    e_num = MetaData[epoch][0]
    jd = MetaData[epoch][4]
    elev = MetaData[epoch][6]
    t225 = MetaData[epoch][7]
    rms = MetaData[epoch][8]
    cal_f = MetaData[epoch][10]
    cal_f_err = MetaData[epoch][11]

    # dictionary [key (distance)] [fit (0) or err (1)] [epoch number] [m (0) or b (1)]

    m5 = Dat_dict[5][0][epoch][0]
    m10 = Dat_dict[10][0][epoch][0]
    m15 = Dat_dict[15][0][epoch][0]

    b5 = Dat_dict[5][0][epoch][1]
    b10 = Dat_dict[10][0][epoch][1]
    b15 = Dat_dict[15][0][epoch][1]

    b5_err = Dat_dict[5][1][epoch][1]
    b10_err = Dat_dict[10][1][epoch][1]
    b15_err = Dat_dict[15][1][epoch][1]
    dat = np.array([e_num, jd, elev, t225, rms, cal_f, cal_f_err,
                    b5, b5_err, m5/b5,
                    b10, b10_err, m10/b10,
                    b15, b15_err, m15/b15], dtype='float')
    li = np.vstack((li, dat))

form = '%d %f %d %g %f %f %f %f %f %f %f %f %f %f %f %f'

np.savetxt('/home/broughtonco/documents/nrc/data/IC348/Datafiles/IC348_M_B_table.table', li[1:], fmt=form, header=hdr)
