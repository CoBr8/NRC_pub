import sys
import numpy as np
import numpy.ma as ma
from itertools import product
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.convolution import interpolate_replace_nans
from astropy.convolution import Gaussian2DKernel
import os as os
import operator as op
from scipy.optimize import curve_fit
from collections import defaultdict
from scipy import odr
from matplotlib import pyplot as plt

# + ===================== +
# | Creating the data     |
# | structure used.       |
# + ===================== +

data = defaultdict(dict)
data['850'] = defaultdict(dict)

data['850']['JCMT_offset'] = defaultdict(dict)  # to use the date as the index

data['850']['dates'] = list()  # to collect all of the dates in the data set

data['850']['linear'] = defaultdict(dict)
data['850']['linear']['m'] = defaultdict(dict)
data['850']['linear']['m_err'] = defaultdict(dict)
data['850']['linear']['b'] = defaultdict(dict)
data['850']['linear']['b_err'] = defaultdict(dict)

data['850']['XC'] = defaultdict(dict)
data['850']['XC']['offset'] = defaultdict(list)
data['850']['XC']['offset_err'] = defaultdict(list)

data['850']['AC'] = defaultdict(dict)
data['850']['AC']['amp'] = defaultdict(list)
data['850']['AC']['amp_err'] = defaultdict(list)
data['850']['AC']['sig_x'] = defaultdict(list)
data['850']['AC']['sig_x_err'] = defaultdict(list)
data['850']['AC']['sig_y'] = defaultdict(list)
data['850']['AC']['sig_y_err'] = defaultdict(list)
data['850']['AC']['theta'] = defaultdict(list)
data['850']['AC']['theta_err'] = defaultdict(list)

data['450'] = defaultdict(dict)

data['450']['JCMT_offset'] = defaultdict(dict)  # to use the date as the index

data['450']['dates'] = list()  # to collect all of the dates in the data set

data['450']['linear'] = defaultdict(list)
data['450']['linear']['m'] = defaultdict(dict)
data['450']['linear']['m_err'] = defaultdict(dict)
data['450']['linear']['b'] = defaultdict(dict)
data['450']['linear']['b_err'] = defaultdict(dict)

data['450']['XC'] = defaultdict(list)
data['450']['XC']['offset'] = defaultdict(list)
data['450']['XC']['offset_err'] = defaultdict(list)

data['450']['AC'] = defaultdict(dict)
data['450']['AC']['amp'] = defaultdict(int)
data['450']['AC']['amp_err'] = defaultdict(int)
data['450']['AC']['sig_x'] = defaultdict(int)
data['450']['AC']['sig_x_err'] = defaultdict(int)
data['450']['AC']['sig_y'] = defaultdict(int)
data['450']['AC']['sig_y_err'] = defaultdict(int)
data['450']['AC']['theta'] = defaultdict(int)
data['450']['AC']['theta_err'] = defaultdict(int)


# + ===================== +
# | Functions used        |
# + ===================== +


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
    :return: y: a quadratic
    """

    y = m * x ** 2 + b
    return y


def f_linear(p, x):
    """
    :param x: independent variable
    :param m: slope of the fitted line
    :paraM b: independent variable intercept
    :return: y: a linear monomial
    """

    y = p[0] * x + p[1]
    return y


# + ===================== +
# | Root project location |
# + ===================== +
# ROOT = '/home/cobr/Documents/jcmt-variability/'
# external disk root location:
ROOT = '/media/cobr/Media Backup1/jcmt-transient/'

# + ===================== +
# | Global parameters     |
# + ===================== +
RADIUS = 7  # the distance used for linear fitting and gaussian fitting (use width = RADIUS*2 + 1)
length = 200  # The size we clip the reference matrix to. size MxM = length*2 x length*2
TEST = False
# + ===================================== +
# | Exception handling, ensuring an arg   |
# | was passed to the file through CLI    |
# + ===================================== +
REGIONS = ['IC348', 'NGC1333', 'NGC2024', 'NGC2071', 'OMC23', 'OPH_CORE', 'SERPENS_MAIN', 'SERPENS_SOUTH']

try:
    regions_inquired = sys.argv[1:]
except IndexError:
    raise Exception("A region must be given as arg. Use -h or --help to view the help string")

if sys.argv[1].lower() in ['-h', '--help']:
    print("""
        ========================================
        This file computes the cross-correlation
        and auto-correlation of all epochs in a
        region.

        To see the parameters of the ascii table
        pass '--table_params'as the first
        argument

        This file was originally designed to use
        for the James Clerk Maxwell Telescope's
        Transient Survey. As such the regions
        supported are as follows:

        one of:
            IC348,        NGC1333,
            NGC2024,      NGC2071,
            OMC23,        OPH_CORE,
            SERPENS_MAIN, SERPENS_SOUTH
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        IN:
            ARG1: Region name

        out:
            ARG1.table
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ========================================""")

for r in regions_inquired:
    if r not in REGIONS:
        raise Exception(
            'The region you provided is not apart of the JCMT_Transient Survey and is not currently supported')
    else:
        regions_inquired = sys.argv[1:]

for region in regions_inquired:
    Dates850 = []
    Dates450 = []
    DataRoot = ROOT + region + "/A3_images/"
    files = os.listdir(DataRoot)  # listing all the files in root

    MetaData850 = np.loadtxt(ROOT + region + '/A3_images_cal/' + region + '_850_EA3_cal_metadata.txt', dtype=str)
    MetaData450 = np.loadtxt(ROOT + region + '/A3_images_cal_450/' + region + '_450_EA3_cal_metadata.txt', dtype=str)

    FN850 = MetaData850.T[1]  # filename of the 850 metadata files (ordered)
    FN450 = MetaData450.T[1]  # filename of the 450 metadata files (ordered)

    Dates850.extend([''.join(d[1:].split('-')) for d in MetaData850.T[2]])  # the dates of all the 850 metadata files
    Dates450.extend([''.join(d[1:].split('-')) for d in MetaData450.T[2]])  # the dates of all the 450 metadata files

    # + =================== +
    # | 450 micron data set |
    # + =================== +
    Files450 = []
    for fn in files:
        if '450' in fn:
            Files450.append(fn)
    FirstEpochName = Files450[0]  # the name of the first epoch
    FirstEpoch = fits.open(DataRoot + '/' + FirstEpochName)
    FirstEpochData = FirstEpoch[0].data[0]  # Numpy data array for the first epoch
    FirstEpochCentre = np.array(
        [FirstEpoch[0].header['CRPIX1'],
         FirstEpoch[0].header['CRPIX2']]
    )  # loc of actual centre

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
    XC_epochs, AC_epochs, DataDates = [], [], []
    dictionary1 = {}
    for fn in files:  # for file_name in files
        FilePath = ROOT + region + "/A3_images/" + fn
        if os.path.isfile(FilePath) and (fn[-4:].lower() == ('.fit' or '.fits')) and ('450' in fn):
            hdul = fits.open(FilePath)  # opening the file in astropy
            date = str(hdul[0].header['UTDATE'])  # extracting the date from the header
            date += '-' + str(hdul[0].header['OBSNUM'])
            centre = (hdul[0].header['CRPIX1'], hdul[0].header['CRPIX2'])  # what does JCMT say the centre of the map is
            Vec = np.array([centre[0] - (hdul[0].shape[2] // 2),
                            centre[1] - (hdul[0].shape[1] // 2)]
                           )
            dictionary1[date] = hdul[0], Vec  # a nice compact way to store the data for later.

    # +===============================================================================+
    # + Creating all of the data needed for the linear fitting and power/radius plots +
    # +===============================================================================+
    for date, (hdu, Vec) in sorted(dictionary1.items(), key=op.itemgetter(0)):  # pulling data from dictionary
        date = str(date)
        Map_of_Region = hdu.data[0]  # map of the region
        JCMT_offset = FirstEpochVec - Vec  # JCMT offset from headers
        data['450']['JCMT_offset'][date] = JCMT_offset  # used for accessing data later.
        Map_of_Region = interpolate_replace_nans(correlate(Map_of_Region, clip_only=True), Gaussian2DKernel(5))
        # using correlate function to clip the map
        XCorr = correlate(epoch_1=Map_of_Region, epoch_2=FirstEpochData).real  # xcorr of epoch with the first
        AC = correlate(Map_of_Region).real  # auto correlation of the map

        AC_epochs.append([date, AC])
        XC_epochs.append([date, XCorr])  # appending to list; used for fitting all maps later
        data['450']['dates'].append(date)
        Clipped_Map_of_Region_LENGTH = np.arange(0, Map_of_Region.shape[0])
        loc = list(product(Clipped_Map_of_Region_LENGTH, Clipped_Map_of_Region_LENGTH))  # all index's of array

        MidMapX = AC.shape[1] // 2  # middle of the map x
        MidMapY = AC.shape[0] // 2  # and y

        radius, AC_pows = [], []

        for idx in loc:  # Determining the power at a certain radius
            r = ((idx[0] - MidMapX) ** 2 + (idx[1] - MidMapY) ** 2) ** (1 / 2)
            AC_pow = AC[idx[0], idx[1]].real
            radius.append(r)
            AC_pows.append(AC_pow)

        DataSetsAC.append(np.array(AC_pows))
        Radius_Data = radius

    # Creating the sorted data for the Power vs Radius Plots:
    AC_Data_450 = []
    for ACDatSet in DataSetsAC:
        r, ACd = zip(*sorted(list(zip(Radius_Data, ACDatSet)), key=op.itemgetter(0)))
        AC_Data_450.append(np.array(ACd))
    r = np.array(r)

    # +================+
    # | Linear fitting |
    # +================+
    """
    For the linear fitting we look at the auto-correlation matrix.
    We start in the middle of the matrix and collect the data points 
    at each radius (r), where r = sqrt(X^2 + Y^2). X and Y are the 
    index positions along the two axis of the matrix. 

    For the linear fitting we take the first set of data points where:
        r <= 7
    We use a function defined as:
        y = m x^2 + b 
    to fit the points as there is a quasi-quadratic curvature in the 
    data. 
    """
    i = 0
    for dist, date in list(zip(np.ones(len(AC_Data_450)) * RADIUS, data['450']['dates'])):
        num = len(r[r <= dist])  # the first "num" data points that correspond to a radius less than "dist"
        opt_fit_AC_450, cov_mat_AC_450 = curve_fit(f, r[1:num], AC_Data_450[i][1:num])
        err_450 = np.sqrt(np.diag(cov_mat_AC_450))

        data['450']['linear']['m'][date] = opt_fit_AC_450[0]
        data['450']['linear']['m_err'][date] = err_450[0]

        data['450']['linear']['b'][date] = opt_fit_AC_450[1]
        data['450']['linear']['b_err'][date] = err_450[1]
        i += 1

    # +============================================================+
    # | Gaussian fitting and offset calculations cross correlation |
    # +============================================================+

    for width in [RADIUS]:
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
            data['450']['XC']['offset'][date] = (X_centre - X_max, Y_centre - Y_max)
            data['450']['XC']['offset_err'][date] = (err[1], err[2])

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
            data['450']['AC']['amp'][date] = float(best_fit_gauss.amplitude.value)
            data['450']['AC']['amp_err'][date] = err[0]
            data['450']['AC']['sig_x'][date] = float(best_fit_gauss.x_stddev.value)
            data['450']['AC']['sig_x_err'][date] = err[3]
            data['450']['AC']['sig_y'][date] = float(best_fit_gauss.y_stddev.value)
            data['450']['AC']['sig_y_err'][date] = err[4]
            data['450']['AC']['theta'][date] = float(best_fit_gauss.theta.value)
            data['450']['AC']['theta_err'][date] = err[5]

    # + =================== +
    # | 850 micron data set |
    # + =================== +
    Files850 = []
    for fn in files:
        if '850' in fn:
            Files850.append(fn)
    FirstEpochName = Files850[0]  # the name of the first epoch
    FirstEpoch = fits.open(DataRoot + '/' + FirstEpochName)
    FirstEpochData = FirstEpoch[0].data[0]  # Numpy data array for the first epoch
    FirstEpochCentre = np.array(
        [FirstEpoch[0].header['CRPIX1'], FirstEpoch[0].header['CRPIX2']])  # loc of actual centre

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
    XC_epochs, AC_epochs, DataDates = [], [], []
    dictionary1 = {}
    for fn in files:  # for file_name in files
        FilePath = ROOT + region + "/A3_images/" + fn
        if os.path.isfile(FilePath) and fn[-4:].lower() == ('.fit' or '.fits') and '850' in fn:
            hdul = fits.open(FilePath)  # opening the file in astropy
            date = str(hdul[0].header['UTDATE'])  # extracting the date from the header
            date += '-' + str(hdul[0].header['OBSNUM'])
            centre = (hdul[0].header['CRPIX1'], hdul[0].header['CRPIX2'])  # what does JCMT say the centre of the map is
            Vec = np.array([centre[0] - (hdul[0].shape[2] // 2),
                            centre[1] - (hdul[0].shape[1] // 2)]
                           )
            dictionary1[date] = hdul[0], Vec  # a nice compact way to store the data for later.
    # +===============================================================================+
    # + Creating all of the data needed for the linear fitting and power/radius plots +
    # +===============================================================================+

    for date, (hdu, Vec) in sorted(dictionary1.items(), key=op.itemgetter(0)):  # pulling data from dictionary
        date = str(date)
        Map_of_Region = hdu.data[0]  # map of the region
        JCMT_offset = FirstEpochVec - Vec  # JCMT offset from headers
        data['850']['JCMT_offset'][date] = JCMT_offset  # used for accessing data later.

        Map_of_Region = correlate(Map_of_Region,
                                  clip_only=True)  # using correlate function to clip the map (defined above)
        XCorr = correlate(epoch_1=Map_of_Region,
                          epoch_2=FirstEpochData).real  # cross correlation of epoch with the first
        AC = correlate(Map_of_Region).real  # auto correlation of the map

        AC_epochs.append([date, AC])
        XC_epochs.append([date, XCorr])  # appending to list; used for fitting all maps later
        data['850']['dates'].append(date)

        Clipped_Map_of_Region_LENGTH = np.arange(0, Map_of_Region.shape[0])
        loc = list(product(Clipped_Map_of_Region_LENGTH, Clipped_Map_of_Region_LENGTH))  # all index's of array

        MidMapX = AC.shape[1] // 2  # middle of the map x
        MidMapY = AC.shape[0] // 2  # and y

        radius, AC_pows = [], []

        for idx in loc:  # Determining the power at a certain radius
            r = ((idx[0] - MidMapX) ** 2 + (idx[1] - MidMapY) ** 2) ** (1 / 2)
            AC_pow = AC[idx[0], idx[1]].real
            # +================================================================+
            radius.append(r)
            AC_pows.append(AC_pow)

        DataSetsAC.append(np.array(AC_pows))
        Radius_Data = radius

    # Creating the sorted data for the Power vs Radius Plots:
    AC_Data_850 = []
    for ACDatSet in DataSetsAC:
        r, ACd = zip(*sorted(list(zip(Radius_Data, ACDatSet)), key=op.itemgetter(0)))
        AC_Data_850.append(np.array(ACd))

    r = np.array(r)

    # +================+
    # | Linear fitting |
    # +================+
    """
    For the linear fitting we look at the auto-correlation matrix.
    We start in the middle of the matrix and collect the data points
    at each radius (r), where r = sqrt(X^2 + Y^2). X and Y are the
    index positions along the two axis of the matrix.

    For the linear fitting we take the first set of data points where:
        r <= 7
    We use a function defined as:
        y = m x^2 + b
    to fit the points as there is a quasi-quadratic curvature in the
    data.
    """
    i = 0
    for dist, date in zip(np.ones(len(AC_Data_850)) * RADIUS, data['850']['dates']):
        num = len(r[r <= dist])  # the first "num" data points that correspond to a radius less than "dist"
        opt_fit_AC, cov_mat_AC = curve_fit(f, r[1:num], AC_Data_850[i][1:num])
        err = np.sqrt(np.diag(cov_mat_AC))

        data['850']['linear']['m'][date] = opt_fit_AC[0]
        data['850']['linear']['m_err'][date] = err[0]

        data['850']['linear']['b'][date] = opt_fit_AC[1]
        data['850']['linear']['b_err'][date] = err[1]
        i += 1

    # +============================================================+
    # | Gaussian fitting and offset calculations cross correlation |
    # +============================================================+
    for width in [RADIUS]:
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
            data['850']['XC']['offset'][date] = (X_centre - X_max, Y_centre - Y_max)
            data['850']['XC']['offset_err'][date] = (err[1], err[2])

        # +==================================+
        # | Autocorrelation Gaussian Fitting |
        # +==================================+
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
            data['850']['AC']['amp'][date] = float(best_fit_gauss.amplitude.value)
            data['850']['AC']['amp_err'][date] = err[0]
            data['850']['AC']['sig_x'][date] = float(best_fit_gauss.x_stddev.value)
            data['850']['AC']['sig_x_err'][date] = err[3]
            data['850']['AC']['sig_y'][date] = float(best_fit_gauss.y_stddev.value)
            data['850']['AC']['sig_y_err'][date] = err[4]
            data['850']['AC']['theta'][date] = float(best_fit_gauss.theta.value)
            data['850']['AC']['theta_err'][date] = err[5]

    # + ===================================== +
    # | Calibration factor via linear fitting |
    # + ===================================== +
    """
    This had to be split into two parts:

    1. For 850um all epochs have cal_f values
    attributed to them

    2. For 450um only some epochs have been
    calibrated, need to know which ones.
    """
    model = odr.Model(f_linear)

    x450 = []
    x450_err = []
    i = 0
    for date450 in data['450']['dates']:
        if str(date450[:8]) in Dates450:
            if data['450']['linear']['m'][date450]>=0:
                NegLinMFLAG = True
            else:
                NegLinMFLAG = False
            x450.append(np.sqrt(np.abs(data['450']['linear']['m'][date450])))
            x450_err.append(0.5 * data['450']['linear']['m_err'][date450] / x450[i])
            i += 1
    cal_f_450 = np.array(MetaData450.T[10], dtype=float)
    cal_f_err_450 = np.array(MetaData450.T[11], dtype=float)
    data450 = odr.RealData(x450, cal_f_450, sx=x450_err, sy=cal_f_err_450)
    odr450 = odr.ODR(data450, model, beta0=[1, 1])
    out450 = odr450.run()
    opt450 = out450.beta
    err450 = out450.sd_beta

    x850 = np.sqrt(-1 * np.array(list(data['850']['linear']['m'].values())))
    x850_err = 0.5 * np.array(list(data['850']['linear']['m_err'].values())) / x850
    cal_f_850 = np.array(MetaData850.T[10], dtype=float)
    cal_f_err_850 = np.array(MetaData850.T[11], dtype=float)

    data850 = odr.RealData(x850, cal_f_850, sx=x850_err, sy=cal_f_err_850)
    odr850 = odr.ODR(data850, model, beta0=[1, 1])
    out850 = odr850.run()
    opt850 = out850.beta
    err850 = out850.sd_beta

    hdr = 'Data_Date Meta_Date File_Name JD Elev T225 RMS Steve_offset_x Steve_offset_y ' \
          'Cal_f Cal_f_err ' \
          'Cal_f_450 Cal_f_450 ' \
          'JCMT_Offset_x JCMT_Offset_y ' \
          'JCMT_Offset_x_450 JCMT_Offset_y_450 ' \
          'XC_off_x XC_off_x_err ' \
          'XC_off_y XC_off_y_err ' \
          'XC_off_x_450 XC_off_x_err_450 ' \
          'XC_off_y_450 XC_off_y_err_450 ' \
          'BA BA_err ' \
          'MA MA_err ' \
          'BA_450 BA_err_450 ' \
          'MA_450 MA_err_450 ' \
          'AC_amp AC_amp_err ' \
          'AC_sig_x AC_sig_x_err ' \
          'AC_sig_y AC_sig_y_err ' \
          'AC_theta AC_theta_err ' \
          'AC_amp_450 AC_amp_err_450 ' \
          'AC_sig_x_450 AC_sig_x_err_450 ' \
          'AC_sig_y_450 AC_sig_y_err_450 ' \
          'AC_theta_450 AC_theta_err_450 ' \
          'dx dy, dx_450 dy_450 ddx ddy ' \
          'x x_err x_450 x_err_450 ' \
          'AC_scale_m AC_scale_m_err AC_scale_b AC_scale_b_err ' \
          'AC_scale_m_450 AC_scale_m_err_450 AC_scale_b_450 AC_scale_b_err_450 '
    li = np.zeros(len(hdr.split()), dtype=str)  # How many columns are in the header above?
    index450 = 0
    index850 = 0
    for date450, date in zip(data['450']['dates'], data['850']['dates']):
        AC_cal_f_m = opt850[0]
        AC_cal_f_m_err = err850[0]
        AC_cal_f_b = opt850[1]
        AC_cal_f_b_err = err850[1]

        AC_cal_f_m_450 = opt450[0]
        AC_cal_f_m_err_450 = err450[0]
        AC_cal_f_b_450 = opt450[1]
        AC_cal_f_b_err_450 = err450[1]

        if str(date450[:8]) in Dates450:
            calf450 = MetaData450[index450][10]
            calferr450 = MetaData450[index450][11]
            x_450 = x450[index450]
            x_err_450 = x450_err[index450]/x_450
            index450 += 1
        else:
            x_450 = -1
            x_err_450 = -1
            calf450 = calferr450 = -1

        if str(date[:8]) in Dates850:
            e_num = str(MetaData850[index850][0])  # index850 number
            metadate = Dates850[index850]
            name = str(MetaData850[index850][1][:-4])  # name of index850
            jd = str(MetaData850[index850][4])  # julian date
            elev = str(MetaData850[index850][6])  # elevation
            t225 = str(MetaData850[index850][7])  # tau-225
            rms = str(MetaData850[index850][8])  # RMS level3
            steve_offset_x = str(MetaData850[index850][-2])
            steve_offset_y = str(MetaData850[index850][-1])
            cal_f = str(MetaData850[index850][10])  # calibration factor from Steve
            cal_f_err = str(MetaData850[index850][11])  # error in calibration factor from Steve
            x = x850[index850]
            x_err = x850_err[index850]/x
            index850 += 1
        else:
            e_num = str(-1)  # index850 number
            metadate = str(-1)
            name = str(-1)  # name of index850
            jd = str(-1)  # julian date
            elev = str(-1)  # elevation
            t225 = str(-1)  # tau-225
            rms = str(-1)  # RMS level
            steve_offset_x = str(-1)
            steve_offset_y = str(-1)
            cal_f = str(-1)  # calibration factor from Steve
            cal_f_err = str(-1)  # error in calibration factor from Steve
            x = -1
            x_err = -1

        jcoffx450 = data['450']['JCMT_offset'][date][0]
        jcoffy450 = data['450']['JCMT_offset'][date][1]

        jcoffx850 = data['850']['JCMT_offset'][date][0]
        jcoffy850 = data['850']['JCMT_offset'][date][1]

        xcoffx450 = data['450']['XC']['offset'][date][0]
        xcoffx450_err = data['450']['XC']['offset_err'][date][0]

        xcoffx850 = data['850']['XC']['offset'][date][0]
        xcoffx850_err = data['850']['XC']['offset_err'][date][0]

        xcoffy450 = data['450']['XC']['offset'][date][1]
        xcoffy450_err = data['450']['XC']['offset_err'][date][1]

        xcoffy850 = data['850']['XC']['offset'][date][1]
        xcoffy850_err = data['850']['XC']['offset_err'][date][1]

        acamp450 = data['450']['AC']['amp'][date]
        acamp450_err = data['450']['AC']['amp_err'][date]

        acamp850 = data['850']['AC']['amp'][date]
        acamp850_err = data['850']['AC']['amp_err'][date]

        acsigx450 = data['450']['AC']['sig_x'][date]
        acsigx450_err = data['450']['AC']['sig_x_err'][date]

        acsigx850 = data['850']['AC']['sig_x'][date]
        acsigx850_err = data['850']['AC']['sig_x_err'][date]

        acsigy450 = data['450']['AC']['sig_y'][date]
        acsigy450_err = data['450']['AC']['sig_y_err'][date]

        acsigy850 = data['850']['AC']['sig_y'][date]
        acsigy850_err = data['850']['AC']['sig_y_err'][date]

        actheta450 = data['450']['AC']['theta'][date]
        actheta450_err = data['450']['AC']['theta_err'][date]

        actheta850 = data['850']['AC']['theta'][date]
        actheta850_err = data['850']['AC']['theta_err'][date]

        b450 = data['450']['linear']['b'][date]
        b450_err = data['450']['linear']['b_err'][date]

        b850 = data['850']['linear']['b'][date]
        b850_err = data['850']['linear']['b_err'][date]

        m450 = data['450']['linear']['m'][date]
        m450_err = data['450']['linear']['m_err'][date]

        m850 = data['850']['linear']['m'][date]
        m850_err = data['850']['linear']['m_err'][date]

        # + ============================================================================ +
        # | the following block of code are for the extra equations requested from Doug: |
        # + ============================================================================ +

        # 450 micron
        dx_450 = (jcoffx450 - xcoffx450) * 2
        dy_450 = (jcoffy450 - xcoffy450) * 2

        # 850 micron
        dx = (jcoffx850 - xcoffx850) * 3
        dy = (jcoffy850 - xcoffy850) * 3
        ddx = dx - dx_450
        ddy = dy - dy_450

        P = np.array(
            [date, metadate, name, jd, elev, t225, rms, steve_offset_x, steve_offset_y,
             cal_f, cal_f_err, calf450, calferr450, jcoffx850, jcoffy850, jcoffx450, jcoffy450,
             xcoffx850, xcoffx850_err, xcoffy850, xcoffy850_err, xcoffx450, xcoffx450_err, xcoffy450, xcoffy450_err,
             b850, b850_err, m850, m850_err, b450, b450_err, m450, m450_err,
             acamp850, acamp850_err, acsigx850, acsigx850_err, acsigy850, acsigy850_err, actheta850, actheta850_err,
             acamp450, acamp450_err, acsigx450, acsigx450_err, acsigy450, acsigy450_err, actheta450, actheta450_err,
             dx, dy, dx_450, dy_450, ddx, ddy,
             x, x_err, x_450, x_err_450,
             AC_cal_f_m, AC_cal_f_m_err, AC_cal_f_b, AC_cal_f_b_err,
             AC_cal_f_m_450, AC_cal_f_m_err_450, AC_cal_f_b_450, AC_cal_f_b_err_450],
            dtype=str)
        li = np.vstack((li, P))
    form = '%s'

    x_fit = np.linspace(min(x850), max(x850), 100)
    y_fit = f_linear(opt850, x_fit)

    x_fit_450 = np.linspace(min(x450), max(x450), 100)
    y_fit_450 = f_linear(opt450, x_fit_450)

    grid = plt.GridSpec(nrows=1, ncols=1)
    fig1 = plt.figure(figsize=(8, 8))
    f1a1 = fig1.add_subplot(grid[0, 0])
    f1a1.set_title(str(region))
    f1a1.set_xlabel('sqrt(-ma)')
    f1a1.set_ylabel('cal_f')
    f1a1.plot(x_fit, y_fit, 'k--', label='best fit')
    f1a1.errorbar(x850, cal_f_850, xerr=x850_err, yerr=cal_f_err_850, label='data', fmt='.', ls='none')

    plt.savefig(ROOT + 'full_cal_test/' + region + '_850.png')
    plt.close()

    fig2 = plt.figure(figsize=(8, 8))
    f2a1 = fig2.add_subplot(grid[0, 0])
    f2a1.set_title(str(region))
    f2a1.set_xlabel('sqrt(-ma)')
    f2a1.set_ylabel('cal_f')
    f2a1.plot(x_fit_450, y_fit_450, 'k--', label='best fit')
    f2a1.errorbar(x450, cal_f_450, xerr=x450_err, yerr=cal_f_err_450, label='data', fmt='.', ls='none')
    plt.savefig(ROOT + 'full_cal_test/' + region + '_450.png')
    plt.close()
    if TEST == True:
        Tf = '_TEST'
    else:
        Tf = ''
    np.savetxt(ROOT + '/full_cal_test/' + region +  Tf + '.table',
               li[1:],
               fmt=form,
               header=hdr
               )
