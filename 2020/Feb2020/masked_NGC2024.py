import numpy as np
import numpy.ma as ma
from itertools import product
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter
import os as os
import operator as op
from scipy.optimize import curve_fit
from collections import defaultdict


# +=================================================+
# + Function Definitions for the scope of this file +
# +=================================================+


def colourbar(mappable):
    """
    :param mappable: a map axes object taken as input to apply a colourbar to

    :return: Edits the figure and subplot to include a colourbar which is scaled to the map correctly
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    figle = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return figle.colorbar(mappable, cax=cax, format='%g')


def correlate(epoch_1=None, epoch_2=None, clipped_side=400, clip_only=False, psd=False):
    """
    :param epoch_1:
        2-Dimensional numpy array. Default: None
        When only epoch_1 is passed it is auto correlated with itself
    :param epoch_2:
        2-Dimensional numpy array. Default: None
        When both epoch_1 and epoch_2 are passed the two arrays are cross correlated
    :param clipped_side:
        Integer. Default: 400.
        The length of one side of the clipped array.
    :param clip_only:
        Boolean. Default: False
        When True is passed to clip_only it will only clip epoch_1
    :param psd:
        Boolean. Default: False
        When true is passed the power spectrum is returned
    :return:
    """
    from numpy.fft import fft2, ifft2, fftshift
    if clip_only:
        mid_map_x, mid_map_y = epoch_1.shape[1] // 2, epoch_1.shape[0] // 2
        clipped_epoch = epoch_1[mid_map_y - clipped_side // 2:mid_map_y + clipped_side // 2 + 1,
                                mid_map_x - clipped_side // 2:mid_map_x + clipped_side // 2 + 1
                                ]
        return clipped_epoch

    elif psd:
        mid_map_x, mid_map_y = epoch_1.shape[1] // 2, epoch_1.shape[0] // 2
        clipped_epoch = epoch_1[mid_map_y - clipped_side // 2:mid_map_y + clipped_side // 2 + 1,
                                mid_map_x - clipped_side // 2:mid_map_x + clipped_side // 2 + 1
                                ]
        psd = fft2(clipped_epoch) * fft2(clipped_epoch).conj()
        return fftshift(psd)

    elif epoch_1 is None:
        raise Exception('You need to pass a 2D map for this function to work')

    elif epoch_2 is None:
        mid_map_x, mid_map_y = epoch_1.shape[1] // 2, epoch_1.shape[0] // 2
        clipped_epoch = epoch_1[mid_map_y - clipped_side // 2:mid_map_y + clipped_side // 2 + 1,
                                mid_map_x - clipped_side // 2:mid_map_x + clipped_side // 2 + 1
                                ]
        ac = ifft2(fft2(clipped_epoch) * fft2(clipped_epoch).conj())
        return fftshift(ac)

    else:
        mid_map_x_1, mid_map_y_1 = epoch_1.shape[1] // 2, epoch_1.shape[0] // 2
        mid_map_x_2, mid_map_y_2 = epoch_2.shape[1] // 2, epoch_2.shape[0] // 2
        clipped_epoch_1 = epoch_1[mid_map_y_1 - clipped_side // 2:mid_map_y_1 + clipped_side // 2 + 1,
                                  mid_map_x_1 - clipped_side // 2:mid_map_x_1 + clipped_side // 2 + 1
                                  ]
        clipped_epoch_2 = epoch_2[mid_map_y_2 - clipped_side // 2:mid_map_y_2 + clipped_side // 2 + 1,
                                  mid_map_x_2 - clipped_side // 2:mid_map_x_2 + clipped_side // 2 + 1
                                  ]
        xcorr = ifft2(fft2(clipped_epoch_1) * fft2(clipped_epoch_2).conj())
        return fftshift(xcorr)


def f(x, m, b):
    """
    :param x: independent variable
    :param m: slope
    :param b: intercept
    :return: a straight line
    """

    y = m * x ** 2 + b
    return y


# +===============================================================================+
# + Creating all of the data needed for the linear fitting and power/radius plots +
# +===============================================================================+
length = 200  # used for clipping the first epoch map

root = '/home/cobr/Documents/NRC/data/NGC2024/850/'  # the root of the data folder
files = os.listdir(root)  # listing all the files in root.

FirstEpoch = fits.open('/home/cobr/Documents/NRC/data/NGC2024/850/NGC2024_20151226_00049_850_EA3_cal.fit')
FirstEpochDate = FirstEpoch[0].header['UTDATE']  # Date of the first epoch
FirstEpochData = FirstEpoch[0].data[0]  # Numpy data array for the first epoch
FirstEpochCentre = np.array([FirstEpoch[0].header['CRPIX1'], FirstEpoch[0].header['CRPIX2']])  # loc of actual centre

# middle of the map of the first epoch
FED_MidMapX = FirstEpochData.shape[1] // 2
FED_MidMapY = FirstEpochData.shape[0] // 2
FirstEpochVec = np.array([FirstEpochCentre[0] - FED_MidMapX,
                          FirstEpochCentre[1] - FED_MidMapY]
                         )

# clipping the map to the correct (400,400) size
# clipping the map to the correct (400,400) size
FirstEpochData = FirstEpochData[
                 FED_MidMapY - length:FED_MidMapY + length + 1,
                 FED_MidMapX - length:FED_MidMapX + length + 1]

DataSetsPSD, DataSetsAC, radiuses = [], [], []
XC_epochs, AC_epochs = [], []
Dat_Titles = ['Map', 'PowerSpectrum', 'AutoCorr', 'XCorr']
dictionary1 = {}
JCMT_offsets = []

# DataSets
for fn in files:  # for file_name in files
    if os.path.isfile(root + '/' + fn) and fn[-4:] != '.txt' and fn[0] != '.':  # files only, but not .txt files or .
        hdul = fits.open(root + '/' + fn)  # opening the file in astropy
        date = hdul[0].header['UTDATE']  # extracting the date from the header
        Region_Name = hdul[0].header['OBJECT']  # what we are looking at
        centre = (hdul[0].header['CRPIX1'], hdul[0].header['CRPIX2'])  # what does JCMT say the centre of the map is
        Vec = np.array([centre[0] - (hdul[0].shape[2] // 2),
                        centre[1] - (hdul[0].shape[1] // 2)]
                       )
        dictionary1[date] = hdul[0], Vec  # a nice compact way to store the data for later.

for date, (hdu, Vec) in sorted(dictionary1.items(), key=op.itemgetter(0)):  # pulling data from dictionary
    Map_of_Region = hdu.data[0]  # map of the region
    Region_Name = hdu.header['OBJECT']  # object in map from fits header
    JCMT_offset = FirstEpochVec - Vec  # JCMT offset from headers
    JCMT_offsets.append(JCMT_offset)  # used for accessing data later.

    Map_of_Region = correlate(Map_of_Region, clip_only=True)  # using correlate function to clip the map (defined above)
    XCorr = correlate(epoch_1=Map_of_Region, epoch_2=FirstEpochData).real  # cross correlation of epoch with the first
    PSD = correlate(Map_of_Region, psd=True).real  # power spectrum of the map
    AC = correlate(Map_of_Region).real  # auto correlation of the map

    AC_epochs.append(AC)
    XC_epochs.append(XCorr)  # appending to list; used for fitting all maps later

    Clipped_Map_of_Region_LENGTH = np.arange(0, Map_of_Region.shape[0])
    loc = list(product(Clipped_Map_of_Region_LENGTH, Clipped_Map_of_Region_LENGTH))  # all index's of array

    MidMapX = AC.shape[1] // 2  # middle of the map x
    MidMapY = AC.shape[0] // 2  # and y

    radius, PSD_pows, AC_pows = [], [], []


    for idx in loc:  # Determining the power at a certain radius
        r = ((idx[0] - MidMapX) ** 2 + (idx[1] - MidMapY) ** 2) ** (1 / 2)
        PSD_pow = PSD[idx[0], idx[1]].real
        AC_pow = AC[idx[0], idx[1]].real
        # +================================================================+
        radius.append(r)
        PSD_pows.append(PSD_pow)
        AC_pows.append(AC_pow)

    DataSetsAC.append(np.array(AC_pows))
    DataSetsPSD.append(np.array(PSD_pows))

Radius_Data = radius

# Creating the sorted data for the Power vs Radius Plots:

AC_Data, PSD_Data = [], []
for ACDatSet, PSDDatSet in zip(DataSetsAC, DataSetsPSD):
    r, ACd = zip(*sorted(list(zip(Radius_Data, ACDatSet)), key=op.itemgetter(0)))
    AC_Data.append(np.array(ACd))

r = np.array(r)

# +================+
# | Linear fitting |
# +================+
lin_fit_dict = {}
for dist in [5, 6, 7]:
    num = len(r[r <= dist])  # the first "num" data points that correspond to a radius less than "dist"
    fit_ac = []
    err_ac = []
    for i in range(len(AC_Data)):
        opt_fit_AC, cov_mat_AC = curve_fit(f, r[1:num], AC_Data[i][1:num])
        fit_ac.append(opt_fit_AC)
        err_ac.append(np.sqrt(np.diag(cov_mat_AC)))
    lin_fit_dict[dist] = [fit_ac, err_ac]

# +============================================================+
# | Gaussian fitting and offset calculations cross correlation |
# +============================================================+
xc_offsets_dict = defaultdict (list)
AC_dat_dict = defaultdict(list)
for width in [5, 6, 7]:
    size = width*2+1
    for XCorr in XC_epochs:
        Y_centre, X_centre = XCorr.shape[0] // 2, XCorr.shape[1] // 2  # centre of the xcorr maps default: (200,200)
        # figuring out where i need to clip to
        Y_max, X_max = np.where(XCorr == XCorr.max())
        Y_max = int(Y_max)
        X_max = int(X_max)

        # clipping map further to better fit a gaussian profile to it
        XCorr = XCorr[Y_max - width:Y_max + width + 1, X_max - width:X_max + width + 1]
        # subtracting half the side to then add the mean values after
        X_max -= XCorr.shape[1] // 2
        Y_max -= XCorr.shape[0] // 2
        # generating the gaussian to fit.

        x_mesh, y_mesh = np.meshgrid(np.arange(XCorr.shape[0]), np.arange(XCorr.shape[1]))
        gauss_init = Gaussian2D(
            amplitude=XCorr.max(),
            x_mean=np.where(XCorr == XCorr.max())[1],  # location to start fitting gaussian
            y_mean=np.where(XCorr == XCorr.max())[0],  # location to start fitting gaussian
            # fixed={},  # any fixed parameters
            bounds={
                'amplitude': (XCorr.max() * 0.90, XCorr.max() * 1.10),
                'x_mean': (int(np.where(XCorr == XCorr.max())[1]) - 1, int(np.where(XCorr == XCorr.max())[1]) + 1),
                'y_mean': (int(np.where(XCorr == XCorr.max())[0]) - 1, int(np.where(XCorr == XCorr.max())[0]) + 1)
                },  # allowing var in amplitude to better fit gauss
        )
        fitting_gauss = LevMarLSQFitter()  # Fitting method; Levenberg-Marquardt Least Squares algorithm
        best_fit_gauss = fitting_gauss(gauss_init, x_mesh, y_mesh, XCorr)  # The best fit for the map
        gauss_model = best_fit_gauss(x_mesh, y_mesh)  # the model itself (if we want to plot it

        # now we can get the location of our peak fitted gaussian and add them back to get a total offset
        Y_max += best_fit_gauss.y_mean.value  # Finding the distance from 0,0 to the centre gaussian
        X_max += best_fit_gauss.x_mean.value  # and y.
        errs = np.sqrt(np.diag(fitting_gauss.fit_info['param_cov']))
        # cross-corr offset calc
        xc_offsets_dict[size].append([(X_centre - X_max, Y_centre - Y_max), (errs[1], errs[2])])

    # +============================================+
    # | Autocorrelation symmetric Gaussian Fitting |
    # +============================================+

    AC_fit_li = []
    for AC in AC_epochs:
        # figuring out where I need to clip to, realistically, this SHOULD be at the physical centre (200,200)
        Y_max, X_max = np.where(AC == AC.max())
        Y_max, X_max = int(Y_max), int(X_max)
        # Setting the middle AC point to be our estimated value of B for a better fit.
        # print(Y_max, X_max)

        mask = np.zeros(AC.shape)
        mask[Y_max, X_max] = 1
        AC_masked = ma.masked_array(AC, mask=mask)

        # clipping map further to better fit a gaussian profile to it
        AC = AC_masked[Y_max - width:Y_max + width + 1, X_max - width:X_max + width + 1]

        # generating the gaussian to fit
        x_mesh, y_mesh = np.meshgrid(np.arange(AC.shape[0]), np.arange(AC.shape[1]))
        gauss_init = Gaussian2D(
            amplitude=AC.max(),
            x_mean=AC.shape[1] // 2,  # location to start fitting gaussian
            y_mean=AC.shape[0] // 2,  # location to start fitting gaussian
            # fixed={'amplitude': True},  # any fixed parameters
            # bounds={},
            # tied={'y_stddev': tie}
        )

        fitting_gauss = LevMarLSQFitter()  # Fitting method; Levenberg-Marquardt Least Squares algorithm
        best_fit_gauss = fitting_gauss(gauss_init, x_mesh, y_mesh, AC)  # The best fit for the map
        # gauss_model = best_fit_gauss(x_mesh, y_mesh)  # the model itself (if we want to plot it

        err = np.sqrt(np.diag(fitting_gauss.fit_info['param_cov']))
        AC_dat_dict[size].append([(float(best_fit_gauss.amplitude.value), err[0]),
                            (float(best_fit_gauss.x_stddev.value), err[3]),
                            (float(best_fit_gauss.y_stddev.value), err[4]),
                            (float(best_fit_gauss.theta.value), err[5])
                            ])

# +=======================+
# | generating the table: |
# +=======================+
# bringing in the Meta data files;
MetaData = np.loadtxt('/home/cobr/Documents/NRC/data/NGC2024/850/NGC2024_850_EA3_cal_metadata.txt', dtype='str')

# header for my table files
hdr = 'Epoch Name JD Elevation T225 RMS Steve_offset_x Steve_offset_y Cal_f Cal_f_err ' \
      'JCMT_Offset_x JCMT_offset_y ' \
      'XC_11_off_x XC_11_Off_x_err XC_11_off_y XC_11_Off_y_err ' \
      'XC_13_off_x XC_13_Off_x_err XC_13_off_y XC_13_Off_y_err ' \
      'XC_15_off_x XC_15_Off_x_err XC_15_off_y XC_15_Off_y_err ' \
      'BA_5 BA_5_err MA_5 MA_5_err ' \
      'BA_6 BA_6_err MA_6 MA_6_err ' \
      'BA_7 BA_7_err MA_7 MA_7_err ' \
      'AC_11_amp AC_11_amp_err AC_11_sig_x AC_11_sig_x_err AC_11_sig_y AC_11_sig_y_err AC_11_theta AC_11_theta_err ' \
      'AC_13_amp AC_13_amp_err AC_13_sig_x AC_13_sig_x_err AC_13_sig_y AC_13_sig_y_err AC_13_theta AC_13_theta_err ' \
      'AC_15_amp AC_15_amp_err AC_15_sig_x AC_15_sig_x_err AC_15_sig_y AC_15_sig_y_err AC_15_theta AC_15_theta_err '

li = np.zeros(len(hdr.split()))  # How many columns are in the header above?

for epoch in range(len(AC_Data)):
    e_num = MetaData[epoch][0]  # epoch number
    name = MetaData[epoch][1]  # name of epoch
    jd = MetaData[epoch][4]  # julian date
    elev = MetaData[epoch][6]  # elevation
    t225 = MetaData[epoch][7]  # tau-225
    rms = MetaData[epoch][8]  # RMS level
    steve_offset = np.array([MetaData[epoch][-2], MetaData[epoch][-1]])
    cal_f = MetaData[epoch][10]  # calibration factor from Steve
    cal_f_err = MetaData[epoch][11]  # error in calibration factor from Steve
    jcmt_offset = JCMT_offsets[epoch]  # the Offset as determined by the centre position of the maps

    xc_11_offset = xc_offsets_dict[11][epoch][0]  # the offset as calculated through gaussian fitting
    xc_11_offset_err = xc_offsets_dict[11][epoch][1]

    xc_13_offset = xc_offsets_dict[13][epoch][0]  # the offset as calculated through gaussian fitting
    xc_13_offset_err = xc_offsets_dict[13][epoch][1]

    xc_15_offset = xc_offsets_dict[15][epoch][0]  # the offset as calculated through gaussian fitting
    xc_15_offset_err = xc_offsets_dict[15][epoch][1]

    AC_11_amp = AC_dat_dict[11][epoch][0][0]  # amplitudes
    AC_11_amp_err = AC_dat_dict[11][epoch][0][1]  # amplitude error
    AC_11_sig_x = AC_dat_dict[11][epoch][1][0]
    AC_11_sig_x_err = AC_dat_dict[11][epoch][1][1]
    AC_11_sig_y = AC_dat_dict[11][epoch][2][0]
    AC_11_sig_y_err = AC_dat_dict[11][epoch][2][1]
    AC_11_theta = AC_dat_dict[11][epoch][3][0] % (2*np.pi)
    AC_11_theta_err = AC_dat_dict[11][epoch][3][1]

    AC_13_amp = AC_dat_dict[13][epoch][0][0]  # amplitudes
    AC_13_amp_err = AC_dat_dict[13][epoch][0][1]  # amplitude error
    AC_13_sig_x = AC_dat_dict[13][epoch][1][0]
    AC_13_sig_x_err = AC_dat_dict[13][epoch][1][1]
    AC_13_sig_y = AC_dat_dict[13][epoch][2][0]
    AC_13_sig_y_err = AC_dat_dict[13][epoch][2][1]
    AC_13_theta = AC_dat_dict[13][epoch][3][0] % (2 * np.pi)
    AC_13_theta_err = AC_dat_dict[13][epoch][3][1]

    AC_15_amp = AC_dat_dict[15][epoch][0][0]  # amplitudes
    AC_15_amp_err = AC_dat_dict[15][epoch][0][1]  # amplitude error
    AC_15_sig_x = AC_dat_dict[15][epoch][1][0]
    AC_15_sig_x_err = AC_dat_dict[15][epoch][1][1]
    AC_15_sig_y = AC_dat_dict[15][epoch][2][0]
    AC_15_sig_y_err = AC_dat_dict[15][epoch][2][1]
    AC_15_theta = AC_dat_dict[15][epoch][3][0] % (2 * np.pi)
    AC_15_theta_err = AC_dat_dict[15][epoch][3][1]

    # dictionary[key][fit (0) or err (1)][epoch number][m (0) or b (1)]
    BA_5 = lin_fit_dict[5][0][epoch][1]
    BA_5_err = lin_fit_dict[5][1][epoch][1]
    MA_5 = lin_fit_dict[5][0][epoch][0]
    MA_5_err = lin_fit_dict[5][1][epoch][0]

    BA_6 = lin_fit_dict[6][0][epoch][1]
    BA_6_err = lin_fit_dict[6][1][epoch][1]
    MA_6 = lin_fit_dict[6][0][epoch][0]
    MA_6_err = lin_fit_dict[6][1][epoch][0]
    
    BA_7 = lin_fit_dict[7][0][epoch][1]
    BA_7_err = lin_fit_dict[7][1][epoch][1]
    MA_7 = lin_fit_dict[7][0][epoch][0]
    MA_7_err = lin_fit_dict[7][1][epoch][0]
    
    dat = np.array([e_num, name, jd, elev, t225, rms, *steve_offset, cal_f, cal_f_err,
                    *jcmt_offset,
                    xc_11_offset[0], xc_11_offset_err[0], xc_11_offset[1], xc_11_offset_err[1],
                    xc_13_offset[0], xc_13_offset_err[0], xc_13_offset[1], xc_13_offset_err[1],
                    xc_15_offset[0], xc_15_offset_err[0], xc_15_offset[1], xc_15_offset_err[1],
                    BA_5, BA_5_err, MA_5, MA_5_err,
                    BA_6, BA_6_err, MA_6, MA_6_err,
                    BA_7, BA_7_err, MA_7, MA_7_err,
                    AC_11_amp, AC_11_amp_err, AC_11_sig_x, AC_11_sig_x_err, AC_11_sig_y, AC_11_sig_y_err, AC_11_theta,
                    AC_11_theta_err,
                    AC_13_amp, AC_13_amp_err, AC_13_sig_x, AC_13_sig_x_err, AC_13_sig_y, AC_13_sig_y_err, AC_13_theta,
                    AC_13_theta_err,
                    AC_15_amp, AC_15_amp_err, AC_15_sig_x, AC_15_sig_x_err, AC_15_sig_y, AC_15_sig_y_err, AC_15_theta,
                    AC_15_theta_err,
                    ]
                   )  # the values for each epoch
    li = np.vstack((li, dat))

# saving the tables
# =================
form = '%s'
np.savetxt('/home/cobr/Documents/NRC/data/NGC2024/850/Datafiles/MASKED_NGC2024.table',
           li[1:],
           fmt=form,
           header=hdr
           )
