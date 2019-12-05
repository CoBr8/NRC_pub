import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter
import os as os
import operator as op
from scipy.optimize import curve_fit


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
        clipped_epoch = epoch_1[mid_map_y - clipped_side // 2:mid_map_y + clipped_side // 2,
                        mid_map_x - clipped_side // 2:mid_map_x + clipped_side // 2
                        ]
        return clipped_epoch

    elif psd:
        mid_map_x, mid_map_y = epoch_1.shape[1] // 2, epoch_1.shape[0] // 2
        clipped_epoch = epoch_1[mid_map_y - clipped_side // 2:mid_map_y + clipped_side // 2,
                        mid_map_x - clipped_side // 2:mid_map_x + clipped_side // 2
                        ]
        psd = fft2(clipped_epoch) * fft2(clipped_epoch).conj()
        return fftshift(psd)

    elif epoch_1 is None:
        raise Exception('You need to pass a 2D map for this function to work')

    elif epoch_2 is None:
        mid_map_x, mid_map_y = epoch_1.shape[1] // 2, epoch_1.shape[0] // 2
        clipped_epoch = epoch_1[mid_map_y - clipped_side // 2:mid_map_y + clipped_side // 2,
                        mid_map_x - clipped_side // 2:mid_map_x + clipped_side // 2
                        ]
        ac = ifft2(fft2(clipped_epoch) * fft2(clipped_epoch).conj())
        return fftshift(ac)

    else:
        mid_map_x_1, mid_map_y_1 = epoch_1.shape[1] // 2, epoch_1.shape[0] // 2
        mid_map_x_2, mid_map_y_2 = epoch_2.shape[1] // 2, epoch_2.shape[0] // 2
        clipped_epoch_1 = epoch_1[mid_map_y_1 - clipped_side // 2:mid_map_y_1 + clipped_side // 2,
                          mid_map_x_1 - clipped_side // 2:mid_map_x_1 + clipped_side // 2
                          ]
        clipped_epoch_2 = epoch_2[mid_map_y_2 - clipped_side // 2:mid_map_y_2 + clipped_side // 2,
                          mid_map_x_2 - clipped_side // 2:mid_map_x_2 + clipped_side // 2
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

root = '/home/broughtonco/documents/nrc/data/OMC23/'  # the root of the data folder
files = os.listdir(root)  # listing all the files in root.

FirstEpoch = fits.open('/home/broughtonco/documents/nrc/data/OMC23/OMC23_20151226_00036_850_EA3_cal.fit')
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
FirstEpochData = FirstEpochData[FED_MidMapY - length:FED_MidMapY + length, FED_MidMapX - length:FED_MidMapX + length]

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

    # writing to a fits file
    dat = [Map_of_Region.real, PSD.real, AC.real, XCorr.real]  # the data we want to write to a fits file
    for data, name in zip(dat, Dat_Titles):  # updating the fits files every time, remove old ones first.
        if os.path.exists('/home/broughtonco/documents/nrc/data/OMC23/GeneratedMapsFits/{}_{}_{}.fit'.format(
                Region_Name, date, name)):
            os.remove('/home/broughtonco/documents/nrc/data/OMC23/GeneratedMapsFits/{}_{}_{}.fit'.format(
                Region_Name, date, name))
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(
            '/home/broughtonco/documents/nrc/data/OMC23/GeneratedMapsFits/{}_{}_{}.fit'.format(Region_Name, date, name))

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

# 1. I've created two new lists to store my data: (NEW_AC_DATA and NEW_PSD_DATA)
AC_Data, PSD_Data = [], []
# 2. Zip the data together with the radius (ie make a duple of the radius point and its corresponding data point)
for ACDatSet, PSDDatSet in zip(DataSetsAC, DataSetsPSD):
    # 3. Sort the duples based on increasing radius
    # 4. Split the zipped data into the radius and data arrays
    r, ACd = zip(*sorted(list(zip(Radius_Data, ACDatSet)), key=op.itemgetter(0)))
    # 5. Redefine my original variables to be this sorted data.
    AC_Data.append(np.array(ACd))
# noinspection PyUnboundLocalVariable
r = np.array(r)

Epoch = 5  # any epoch would work, 5 was the lucky number to plot
row, col = 4, 4  # used for GridSpec to create a well gridded figure.

# bringing in the Meta data files;
MetaData = np.loadtxt('/home/broughtonco/documents/nrc/data/OMC23/OMC23_850_EA3_cal_metadata.txt', dtype='str')
BaseDate = MetaData[0][2][1:]  # the base epoch date
Date = MetaData[Epoch - 1][2][1:]  # the used epoch date
Dat_dict_div = {}  # dictionaries because efficient
Dat_dict = {}
dist: int = 5  # radius we are interested in.
num = len(r[r <= dist])  # the first "num" data points that correspond to a radius less than "dist"
DivACData = []
fit_ac_div, cov_ac_div = [], []
err_ac_div = []
fit_ac, cov_ac = [], []
err_ac = []

for i in range(len(AC_Data)):
    DivAC = AC_Data[i] / AC_Data[0]
    opt_fit_AC, cov_mat_AC = curve_fit(f, r[1:num], AC_Data[i][1:num])
    opt_fit_AC_div, cov_mat_AC_div = curve_fit(f, r[1:num], DivAC[1:num])

    DivACData.append(DivAC)
    fit_ac.append(opt_fit_AC)
    cov_ac.append(cov_mat_AC)
    err_ac.append(np.sqrt(np.diag(cov_mat_AC)))

    fit_ac_div.append(opt_fit_AC_div)
    cov_ac_div.append(cov_mat_AC_div)
    err_ac_div.append(np.sqrt(np.diag(cov_mat_AC_div)))

Dat_dict_div[dist] = [fit_ac_div, err_ac_div]
Dat_dict[dist] = [fit_ac, err_ac]

# Gaussian fitting and offset calculations cross correlation
# +========================================================+
xc_offsets = []
xc_offsets_errs = []
for XCorr in XC_epochs:
    Y_centre, X_centre = XCorr.shape[0] // 2, XCorr.shape[1] // 2  # centre of the xcorr maps default: (200,200)

    # figuring out where i need to clip to
    Y_max, X_max = np.where(XCorr == XCorr.max())
    Y_max = int(Y_max)
    X_max = int(X_max)

    # clipping map further to better fit a gaussian profile to it
    XCorr = XCorr[Y_max - 5:Y_max + 6, X_max - 5:X_max + 6]
    # subtracting half the side to then add the mean values after
    X_max -= XCorr.shape[1] // 2
    Y_max -= XCorr.shape[0] // 2
    # generating the gaussian to fit.

    x_mesh, y_mesh = np.meshgrid(np.arange(XCorr.shape[0]), np.arange(XCorr.shape[1]))
    gauss_init = Gaussian2D(
        amplitude=XCorr.max(),
        x_mean=np.where(XCorr == XCorr.max())[1],  # location to start fitting gaussian
        y_mean=np.where(XCorr == XCorr.max())[0],  # location to start fitting gaussian
        fixed={},  # any fixed parameters
        bounds={
            'amplitude': (XCorr.max() * 0.90, XCorr.max() * 1.10),
            # 'x_mean': (int(np.where(XCorr == XCorr.max())[1]) - 1, int(np.where(XCorr == XCorr.max())[1]) + 1),
            # 'y_mean': (int(np.where(XCorr == XCorr.max())[0]) - 1, int(np.where(XCorr == XCorr.max())[0]) + 1)
            },  # allowing var in amplitude to better fit gauss
    )
    fitting_gauss = LevMarLSQFitter()  # Fitting method; Levenberg-Marquardt Least Squares algorithm
    best_fit_gauss = fitting_gauss(gauss_init, x_mesh, y_mesh, XCorr)  # The best fit for the map
    gauss_model = best_fit_gauss(x_mesh, y_mesh)  # the model itself (if we want to plot it

    # now we can get the location of our peak fitted gaussian and add them back to get a total offset
    Y_max += best_fit_gauss.y_mean.value  # Finding the distance from 0,0 to the centre gaussian
    X_max += best_fit_gauss.x_mean.value  # and y.

    xc_offsets.append((X_centre - X_max, Y_centre - Y_max))  # cross-corr offset calc

    xc_offsets_errs.append(
        (
            np.sqrt(np.diag(fitting_gauss.fit_info['param_cov'])[1]),
            np.sqrt(np.diag(fitting_gauss.fit_info['param_cov'])[2])
        )
    )  # error in calculations


def tie(m):
    ret = m.x_stddev
    return ret















# Autocorrelation symmetric Gaussian Fitting
# +========================================+
AC_fit_li = []
for i, AC in enumerate(AC_epochs):
    B = Dat_dict[5][0][i][1]
    # figuring out where i need to clip to
    Y_max, X_max = np.where(AC == AC.max())
    Y_max, X_max = int(Y_max), int(X_max)
    # Setting the middle AC point to be our estimated value of B for a better fit.
    AC[Y_max, X_max] = B

    # clipping map further to better fit a gaussian profile to it
    AC = AC[Y_max - 5:Y_max + 6, X_max - 5:X_max + 6]

    # generating the gaussian to fit.
    x_mesh, y_mesh = np.meshgrid(np.arange(AC.shape[0]), np.arange(AC.shape[1]))
    gauss_init = Gaussian2D(
        amplitude=AC.max(),
        x_mean=AC.shape[1] // 2,  # location to start fitting gaussian
        y_mean=AC.shape[0] // 2,  # location to start fitting gaussian
        # fixed={'amplitude': True},  # any fixed parameters
        bounds={},
        # tied={'y_stddev': tie}
    )

    fitting_gauss = LevMarLSQFitter()  # Fitting method; Levenberg-Marquardt Least Squares algorithm
    best_fit_gauss = fitting_gauss(gauss_init, x_mesh, y_mesh, AC)  # The best fit for the map
    gauss_model = best_fit_gauss(x_mesh, y_mesh)  # the model itself (if we want to plot it

    err = np.sqrt(np.diag(fitting_gauss.fit_info['param_cov']))
    AC_fit_li.append(((float(best_fit_gauss.amplitude.value), err[0]),
                      (float(best_fit_gauss.x_stddev.value), err[3]),
                      (float(best_fit_gauss.y_stddev.value), err[4])
                      ))
    i+=1















# generating the table:
# +===================+
# header for my table files
hdr = 'Epoch Name JD Elevation T225 RMS Steve_offset_x Steve_offset_y Cal_f Cal_f_err ' \
      'BA BA_err MA MA_err ' \
      'JCMT_Offset_x JCMT_offset_y XCorr_offset_x XCorr_Offset_x_err XCorr_offset_y XCorr_Offset_y_err ' \
      'BD BD_err MD MD_err ' \
      'AC_amp AC_amp_err AC_sig_x AC_sig_x_err AC_sig_y AC_sig_y_err'
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
    xc_offset = xc_offsets[epoch]  # the offset as calculated through gaussian fitting
    xc_offset_err = xc_offsets_errs[epoch]

    AC_amp = AC_fit_li[epoch][0][0]  # amplitudes
    AC_amp_err = AC_fit_li[epoch][0][1]  # amplitude error
    AC_sig_x = AC_fit_li[epoch][1][0]
    AC_sig_x_err = AC_fit_li[epoch][1][1]
    AC_sig_y = AC_fit_li[epoch][2][0]
    AC_sig_y_err = AC_fit_li[epoch][2][0]

    # dictionary[key][fit (0) or err (1)][epoch number][m (0) or b (1)]
    BA = Dat_dict[5][0][epoch][1]
    BA_err = Dat_dict[5][1][epoch][1]
    MA = Dat_dict[5][0][epoch][0]
    MA_err = Dat_dict[5][1][epoch][0]

    BD = Dat_dict_div[5][0][epoch][1]
    BD_err = Dat_dict_div[5][1][epoch][1]
    MD = Dat_dict_div[5][0][epoch][0]
    MD_err = Dat_dict_div[5][1][epoch][0]

    dat = np.array([e_num, name, jd, elev, t225, rms, *steve_offset, cal_f, cal_f_err,
                    BA, BA_err, MA, MA_err,
                    *jcmt_offset,
                    xc_offset[0][0], xc_offset_err[0],
                    xc_offset[1][0], xc_offset_err[1],
                    BD, BD_err, MD, MD_err,
                    AC_amp, AC_amp_err, AC_sig_x, AC_sig_x_err, AC_sig_y, AC_sig_y_err]
                   )  # the values for each epoch
    li = np.vstack((li, dat))

# saving the tables in two formats, human readable and TOPCAT use able
form = '%s'
np.savetxt('/home/broughtonco/documents/nrc/data/OMC23/Datafiles/OMC23_TABLE_TOPCAT.table',
           li[1:],
           fmt=form,
           header=hdr
           )
