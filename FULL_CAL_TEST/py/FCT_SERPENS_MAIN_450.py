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

root = '/home/cobr/Documents/NRC/FULL_CAL_TEST/SERPENS_MAIN/450/'  # the root of the data folder
files = os.listdir(root)  # listing all the files in root.

FirstEpoch = fits.open('/home/cobr/Documents/NRC/FULL_CAL_TEST/SERPENS_MAIN/450/SERPENS_MAIN_20160202_00054_450_EA3.fit')
FirstEpochDate = FirstEpoch[0].header['UTDATE']  # Date of the first epoch
FirstEpochData = FirstEpoch[0].data[0]  # Numpy data array for the first epoch
FirstEpochCentre = np.array([FirstEpoch[0].header['CRPIX1'], FirstEpoch[0].header['CRPIX2']])  # loc of actual centre

# middle of the map of the first epoch
FED_MidMapX = FirstEpochData.shape[1] // 2
FED_MidMapY = FirstEpochData.shape[0] // 2
FirstEpochVec = np.array([FirstEpochCentre[0] - FED_MidMapX,
                          FirstEpochCentre[1] - FED_MidMapY]
                         )

# clipping the map to the correct (401,401) size
# clipping the map to the correct (401,401) size
FirstEpochData = FirstEpochData[
                 FED_MidMapY - length:FED_MidMapY + length + 1,
                 FED_MidMapX - length:FED_MidMapX + length + 1]

DataSetsPSD, DataSetsAC, radiuses = [], [], []
XC_epochs, AC_epochs, epoch_Dates = [], [], []
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
    AC = correlate(Map_of_Region).real  # auto correlation of the map

    AC_epochs.append([date, AC])
    XC_epochs.append([date, XCorr])  # appending to list; used for fitting all maps later
    epoch_Dates.append(date)

    Clipped_Map_of_Region_LENGTH = np.arange(0, Map_of_Region.shape[0])
    loc = list(product(Clipped_Map_of_Region_LENGTH, Clipped_Map_of_Region_LENGTH))  # all index's of array

    MidMapX = AC.shape[1] // 2  # middle of the map x
    MidMapY = AC.shape[0] // 2  # and y

    radius, PSD_pows, AC_pows = [], [], []

    for idx in loc:  # Determining the power at a certain radius
        r = ((idx[0] - MidMapX) ** 2 + (idx[1] - MidMapY) ** 2) ** (1 / 2)
        AC_pow = AC[idx[0], idx[1]].real
        # +================================================================+
        radius.append(r)
        AC_pows.append(AC_pow)

    DataSetsAC.append(np.array(AC_pows))
Radius_Data = radius

# Creating the sorted data for the Power vs Radius Plots:

AC_Data, PSD_Data = [], []
for ACDatSet in DataSetsAC:
    r, ACd = zip(*sorted(list(zip(Radius_Data, ACDatSet)), key=op.itemgetter(0)))
    AC_Data.append(np.array(ACd))

r = np.array(r)

# +================+
# | Linear fitting |
# +================+
lin_fit_dict = {}
for dist in [7]:
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
xc_offsets_dict = defaultdict(list)
AC_dat_dict = defaultdict(list)
for width in [7]:
    size = width * 2 + 1
    for date, XCorr in XC_epochs:
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
                # 'amplitude': (XCorr.max() * 0.90, XCorr.max() * 1.10),
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
        try:
            err = np.sqrt(np.diag(fitting_gauss.fit_info['param_cov']))
        except:
            err = np.ones(10) * -5
        # cross-corr offset calc
        xc_offsets_dict[date].append([(X_centre - X_max, Y_centre - Y_max), (err[1], err[2])])

    # +============================================+
    # | Autocorrelation symmetric Gaussian Fitting |
    # +============================================+

    AC_fit_li = []
    for date, AC in AC_epochs:
        # figuring out where I need to clip to, realistically, this SHOULD be at the physical centre (200,200)
        Y_max, X_max = np.where(AC == AC.max())
        Y_max, X_max = int(Y_max), int(X_max)
        # Setting the middle AC point to be our estimated value of B for a better fit.

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
        )

        fitting_gauss = LevMarLSQFitter()  # Fitting method; Levenberg-Marquardt Least Squares algorithm
        best_fit_gauss = fitting_gauss(gauss_init, x_mesh, y_mesh, AC)  # The best fit for the map
        # gauss_model = best_fit_gauss(x_mesh, y_mesh)  # the model itself (if we want to plot it
        try:
            err = np.sqrt(np.diag(fitting_gauss.fit_info['param_cov']))
        except:
            err = np.ones(10) * -5
        AC_dat_dict[date].append(
            [
                (float(best_fit_gauss.amplitude.value), err[0]),
                (float(best_fit_gauss.x_stddev.value), err[3]),
                (float(best_fit_gauss.y_stddev.value), err[4]),
                (float(best_fit_gauss.theta.value), err[5])
            ]
        )
# +=======================+
# | generating the table: |
# +=======================+
# bringing in the Meta data files;
MetaData = np.loadtxt('/home/cobr/Documents/NRC/data/SERPENS_MAIN/450/SERPENS_MAIN_450_EA3_cal_metadata.txt', dtype='str')
meta_dates = []
for meta in MetaData:
    metadate = meta[1][-18:-10]
    meta_dates.append(metadate)

# header for my table files
hdr = 'Epoch Name JD Elevation T225 RMS Steve_offset_x Steve_offset_y Cal_f Cal_f_err ' \
      'JCMT_Offset_x JCMT_offset_y ' \
      'XC_off_x XC_Off_x_err XC_off_y XC_Off_y_err ' \
      'BA BA_err MA MA_err ' \
      'AC_amp AC_amp_err AC_sig_x AC_sig_x_err AC_sig_y AC_sig_y_err AC_theta AC_theta_err '

li = np.zeros(len(hdr.split()), dtype=str)  # How many columns are in the header above?
epoch = 0
index = 0
for date in epoch_Dates:
    if str(date) in meta_dates:
        e_num = str(MetaData[epoch][0])  # epoch number
        name = str(MetaData[epoch][1])  # name of epoch
        jd = str(MetaData[epoch][4])  # julian date
        elev = str(MetaData[epoch][6])  # elevation
        t225 = str(MetaData[epoch][7])  # tau-225
        rms = str(MetaData[epoch][8])  # RMS level
        steve_offset_x = str(MetaData[epoch][-2])
        steve_offset_y = str(MetaData[epoch][-1])
        cal_f = str(MetaData[epoch][10])  # calibration factor from Steve
        cal_f_err = str(MetaData[epoch][11])  # error in calibration factor from Steve
        epoch += 1
    else:
        e_num = str(-1)
        name = str(date)  # name of epoch
        jd = str(-1)  # julian date
        elev = str(-1)  # elevation
        t225 = str(-1)  # tau-225
        rms = str(-1)  # RMS level
        steve_offset_x = str(-1)
        steve_offset_y = str(-1)
        cal_f = str(-1)  # calibration factor from Steve
        cal_f_err = str(-1)  # error in calibration factor from Steve

    jcmt_offset_x = JCMT_offsets[index][0]  # the Offset as determined by the centre position of the maps
    jcmt_offset_y = JCMT_offsets[index][1]
    xc_offset_x = str(xc_offsets_dict[date][0][0][0])  # the offset as calculated through gaussian fitting
    xc_offset_x_err = str(xc_offsets_dict[date][0][1][0])
    xc_offset_y = str(xc_offsets_dict[date][0][0][1])
    xc_offset_y_err = str(xc_offsets_dict[date][0][1][1])
    AC_amp = str(AC_dat_dict[date][0][0][0])  # amplitudes
    AC_amp_err = str(AC_dat_dict[date][0][0][1])  # amplitude error
    AC_sig_x = str(AC_dat_dict[date][0][1][0])
    AC_sig_x_err = str(AC_dat_dict[date][0][1][1])
    AC_sig_y = str(AC_dat_dict[date][0][2][0])
    AC_sig_y_err = str(AC_dat_dict[date][0][2][1])
    AC_theta = str(AC_dat_dict[date][0][3][0] % (2 * np.pi))
    AC_theta_err = str(AC_dat_dict[date][0][3][1])
    BA = str(lin_fit_dict[7][0][index][1])
    BA_err = str(lin_fit_dict[7][1][index][1])
    MA = str(lin_fit_dict[7][0][index][0])
    MA_err = str(lin_fit_dict[7][1][index][0])
    data = [e_num, name, jd, elev, t225, rms, steve_offset_x, steve_offset_y, cal_f, cal_f_err,
            jcmt_offset_x, jcmt_offset_y,
            xc_offset_x, xc_offset_x_err, xc_offset_y, xc_offset_y_err, BA, BA_err, MA, MA_err,
            AC_amp, AC_amp_err, AC_sig_x, AC_sig_x_err, AC_sig_y, AC_sig_y_err, AC_theta, AC_theta_err]
    blyat = np.array(data, dtype=str)
    # print(data)
    li = np.vstack((li, blyat))

    index += 1

# saving the tables
# =================
form = '%s'
np.savetxt('/home/cobr/Documents/NRC/FULL_CAL_TEST/TABLES/FCT_SERPENS_MAIN_450.table',
           li[1:],
           fmt=form,
           header=hdr
           )
