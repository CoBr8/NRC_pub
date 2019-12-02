import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
from astropy.modeling.fitting import LevMarLSQFitter
import os as os
import operator as op
from scipy.optimize import curve_fit


def colourbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    figle = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return figle.colorbar(mappable, cax=cax, format='%g')


def correlate(epoch_1=None, epoch_2=None, clipped_side=400, clip_only=False, psd=False):
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
        ac = ifft2(fft2(clipped_epoch)*fft2(clipped_epoch).conj())
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
        xcorr = ifft2(fft2(clipped_epoch_1)*fft2(clipped_epoch_2).conj())
        return fftshift(xcorr)


def f(x, m, b):
    y = m * x + b
    return y


length = 200

root = '/home/broughtonco/documents/nrc/data/OMC23/'
files = os.listdir(root)

FirstEpoch = fits.open('/home/broughtonco/documents/nrc/data/OMC23/OMC23_20151226_00036_850_EA3_cal.fit')
FirstEpochDate = FirstEpoch[0].header['UTDATE']
FirstEpochData = FirstEpoch[0].data[0]
FirstEpochCentre = (FirstEpoch[0].header['CRPIX1'], FirstEpoch[0].header['CRPIX2'])
FED_MidMapX = FirstEpochData.shape[1] // 2
FED_MidMapY = FirstEpochData.shape[0] // 2

FirstEpochData = FirstEpochData[FED_MidMapY-length:FED_MidMapY+length, FED_MidMapX-length:FED_MidMapX+length]

DataSetsPSD, DataSetsAC, radiuses = [], [], []
XC_epochs = []
Dat_Titles = ['Map', 'PowerSpectrum', 'AutoCorr', 'XCorr']
dictionary1 = {}
JCMT_offsets = []

# DataSets
for fn in files:
    if os.path.isfile(root+'/'+fn) and fn[-4:] != '.txt':
        hdul = fits.open(root+'/'+fn)
        date = hdul[0].header['UTDATE']
        Region_Name = hdul[0].header['OBJECT']
        centre = (hdul[0].header['CRPIX1'], hdul[0].header['CRPIX2'])
        dictionary1[date] = hdul[0], centre

for date, (hdu, centre) in sorted(dictionary1.items(), key=op.itemgetter(0)):
    Map_of_Region = hdu.data[0]
    Region_Name = hdu.header['OBJECT']
    JCMT_offset = (FirstEpochCentre[0] - centre[0], FirstEpochCentre[1] - centre[1])
    JCMT_offsets.append(JCMT_offset)

    Map_of_Region = correlate(Map_of_Region, clip_only=True)
    XCorr = correlate(epoch_1=Map_of_Region, epoch_2=FirstEpochData).real
    PSD = correlate(Map_of_Region, psd=True).real
    AC = correlate(Map_of_Region).real

    XC_epochs.append(XCorr)

    Clipped_Map_of_Region_LENGTH = np.arange(0, Map_of_Region.shape[0])
    loc = list(product(Clipped_Map_of_Region_LENGTH, Clipped_Map_of_Region_LENGTH))

    MidMapX = AC.shape[1] // 2
    MidMapY = AC.shape[0] // 2

    radius, PSD_pows, AC_pows = [], [], []

    # writing to a fits file
    dat = [Map_of_Region.real, PSD.real, AC.real, XCorr.real]

    for data, name in zip(dat, Dat_Titles):
        if os.path.exists('/home/broughtonco/documents/nrc/data/OMC23/GeneratedMapsFits/{}_{}_{}.fit'.format(
                Region_Name, date, name)):
            os.remove('/home/broughtonco/documents/nrc/data/OMC23/GeneratedMapsFits/{}_{}_{}.fit'.format(
                Region_Name, date, name))
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(
            '/home/broughtonco/documents/nrc/data/OMC23/GeneratedMapsFits/{}_{}_{}.fit'.format(Region_Name, date, name))

    for idx in loc:
        r = ((idx[0] - MidMapX) ** 2 + (idx[1] - MidMapY) ** 2) ** (1 / 2)
        PSD_pow = PSD[idx[0], idx[1]].real
        AC_pow = AC[idx[0], idx[1]].real
        radius.append(r)
        PSD_pows.append(PSD_pow)
        AC_pows.append(AC_pow)
    DataSetsAC.append(np.array(AC_pows))
    DataSetsPSD.append(np.array(PSD_pows))

    AC[AC.shape[0]//2, AC.shape[1]//2] = 0

    fig1 = plt.figure(figsize=(20, 20))
    grid = plt.GridSpec(5, 4, hspace=0.4, wspace=0.2)

    # row 1

    MapIM = fig1.add_subplot(grid[0, 0])
    acIM = fig1.add_subplot(grid[0, 1])
    psdIM = fig1.add_subplot(grid[0, 2])
    xcorrIM = fig1.add_subplot(grid[0, 3])

    # row 2 + 3

    AvgScatterPSD = fig1.add_subplot(grid[1:3, :])

    # row 3 + 4

    AvgScatterAC = fig1.add_subplot(grid[3:5, :])

    acIM.get_shared_x_axes().join(acIM, psdIM)
    acIM.get_shared_y_axes().join(acIM, psdIM)

    # row 1

    MapIM.set_title('Map')
    im1 = MapIM.imshow(Map_of_Region, origin='lower', cmap='magma')
    colourbar(im1)

    acIM.set_title('Auto-correlation')
    im2 = acIM.imshow(AC, origin='lower', cmap='magma')
    colourbar(im2)

    psdIM.set_title('Power Spectrum')
    im3 = psdIM.imshow(PSD, origin='lower', cmap='magma')
    colourbar(im3)

    xcorrIM.set_title('X-Correlation')
    im4 = xcorrIM.imshow(XCorr, origin='lower', cmap='magma')
    colourbar(im4)

    # row 3

    AvgScatterPSD.set_title('Dispersion of Signal power at a a given frequency')
    AvgScatterPSD.set_xlabel('Frequency')
    AvgScatterPSD.set_ylabel('Power')
    AvgScatterPSD.set_yscale('log')
    AvgScatterPSD.set_ylim(bottom=10**-3, top=10**9)
    im5 = AvgScatterPSD.scatter(radius, PSD_pows, marker=',', lw=0, alpha=0.3, color='red', s=1)

    AvgScatterAC.set_title('Auto Correlation')
    AvgScatterAC.set_xlabel('place holder')
    AvgScatterAC.set_ylabel('p-h')
    AvgScatterAC.set_yscale('log')
    AvgScatterAC.set_aspect('auto')
    AvgScatterAC.set_ylim(bottom=10**-2, top=10**6)
    im6 = AvgScatterAC.scatter(radius, AC_pows, marker=',', lw=0, alpha=0.3, color='red', s=1)

    fig1.suptitle('{} OMC23 epoch'.format(date))
    # plt.show()
    fig1.savefig('/home/broughtonco/documents/nrc/data/OMC23/OMC23_plots/epochs/{}'.format(date))

xc_offsets = []
# Gaussian fitting and offset calculations:
for XCorr in XC_epochs:
    Y_centre, X_centre = XCorr.shape[0]//2, XCorr.shape[1]//2
    XCorr_fitted_offset_Y, XCorr_fitted_offset_X = XCorr.shape[0]//2, XCorr.shape[1]//2
    # need to reshape the map again to better fit the gaussian.
    XCorr = correlate(XCorr, clipped_side=30, clip_only=True)
    # subtract half the map to get
    XCorr_fitted_offset_X -= XCorr.shape[1] // 2
    XCorr_fitted_offset_Y -= XCorr.shape[0] // 2

    x_mesh, y_mesh = np.meshgrid(np.arange(XCorr.shape[0]), np.arange(XCorr.shape[1]))
    gauss_init = Gaussian2D(
        # amplitude=XCorr.max(),
        x_mean=np.where(XCorr == XCorr.max())[1],
        y_mean=np.where(XCorr == XCorr.max())[0],
        fixed={},
        bounds={'amplitude': (XCorr.max() * 0.90, XCorr.max() * 1.10)},
    )
    fitting_gauss = LevMarLSQFitter()
    best_fit_gauss = fitting_gauss(gauss_init, x_mesh, y_mesh, XCorr)
    # cov_diag = np.diag(fitting_gauss.fit_info['param_cov'])
    gauss_model = best_fit_gauss(x_mesh, y_mesh)
    # now we can get the location of our peak fitted gaussian and add them back to get a total offset

    XCorr_fitted_offset_Y += best_fit_gauss.y_mean.value
    XCorr_fitted_offset_X += best_fit_gauss.x_mean.value
    xc_offsets.append((X_centre-XCorr_fitted_offset_X, Y_centre - XCorr_fitted_offset_Y))
# noinspection PyUnboundLocalVariable
Radius_Data = radius

# for the sake of me not forgetting what I've done here I'm actually going to comment here
# I've created two new lists to store my data, (NEW_AC_DATA and NEW_PSD_DATA)
# I then zip the data together with the radius (ie make a duple of the radius point with its corresponding data point)
# then I sort the duples based on the radius ie. a lower radius at the beginning and a larger at the end
# i then split the zipped data into the corresponding radius and data arrays
# from there I redefine my original variables to be this sorted data.

NEW_AC_DATA, NEW_PSD_DATA = [], []
for ACDatSet, PSDDatSet in zip(DataSetsAC, DataSetsPSD):
    r, ACd = zip(*sorted(list(zip(Radius_Data, ACDatSet)), key=op.itemgetter(0)))
    NEW_AC_DATA.append(np.array(ACd))
# noinspection PyUnboundLocalVariable
r = np.array(r)
AC_Data = NEW_AC_DATA  # observed power at a radius from (200, 200) sorted by radius. 0 -> edge
r2 = r ** 2

Epoch = 5
row, col = 4, 4

MetaData = np.loadtxt('/home/broughtonco/documents/nrc/data/OMC23/OMC23_850_EA3_cal_metadata.txt', dtype='str')

BaseDate = MetaData[0][2][1:]
Date = MetaData[Epoch - 1][2][1:]

hdr = 'Epoch JD Elevation T225 RMS Cal_f Cal_f_err JCMT_Offset_x JCMT_offset_y ' \
      'B_5 B_5_err m_5 m_5_err Covariance XCorr_offset_x, XCorr_offset_y'

li = np.zeros(16)
Dat_dict = {}
dist: int = 5
# noinspection PyTypeChecker
num = len(r[r <= dist])

DivACData, DivPSDData = [], []
fit_ac, cov_ac = [], []
err_ac = []

for i in range(len(AC_Data)):
    DivAC = AC_Data[i] / AC_Data[0]
    DivACData.append(DivAC)
    optimal_fit_AC, cov_mat_AC = curve_fit(f, r2[1:num], DivAC[1:num])
    fit_ac.append(optimal_fit_AC)
    cov_ac.append(cov_mat_AC)
    err_ac.append(np.sqrt(np.diag(cov_mat_AC)))

Dat_dict[dist] = [fit_ac, err_ac]
# # Plotting follows
# #
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
fig.savefig('/home/broughtonco/documents/nrc/data/OMC23/OMC23_plots/OMC23_linearFit_radius_{}.png'.format(dist))
# plt.show()

for epoch in range(len(AC_Data)):
    e_num = MetaData[epoch][0]
    jd = MetaData[epoch][4]
    elev = MetaData[epoch][6]
    t225 = MetaData[epoch][7]
    rms = MetaData[epoch][8]
    cal_f = MetaData[epoch][10]
    cal_f_err = MetaData[epoch][11]
    jcmt_offset = JCMT_offsets[epoch]
    xc_offset = xc_offsets[epoch]
    # dictionary[key][fit (0) or err (1)][epoch number][m (0) or b (1)]
    b5 = Dat_dict[5][0][epoch][1]
    b5_err = Dat_dict[5][1][epoch][1]
    m5 = Dat_dict[5][0][epoch][0]
    m5_err = Dat_dict[5][1][epoch][0]
    covariance = cov_ac[epoch][0, 1]
    dat = np.array([e_num, jd, elev, t225, rms, cal_f, cal_f_err, *jcmt_offset,
                    b5, b5_err, m5, m5_err, covariance, *xc_offset],
                   dtype='float'
                   )
    li = np.vstack((li, dat))

# 'Epoch JD Elevation T225 RMS Cal_f Cal_f_err JCMT_Offset B_5 B_5_err m_5 m_5_err Covariance'

frmt = '% 3d % 14f % 4d % 5f % 8f % 8f % 8f % 3d % 3d % 8f % 8f % 8f % 8f % 8f % 8.4f % 8.4f'
form = '%s'

np.savetxt('/home/broughtonco/documents/nrc/data/OMC23/Datafiles/OMC23_TABLE_TOPCAT.table',
           li[1:],
           fmt=form,
           header=hdr
           )

np.savetxt('/home/broughtonco/documents/nrc/data/OMC23/Datafiles/OMC23_TABLE_READABLE.table',
           li[1:],
           fmt=frmt,
           header=hdr
           )
