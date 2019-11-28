import numpy as np
from numpy.fft import ifft2, fftshift
import matplotlib.pyplot as plt
from itertools import product
from astropy.io import fits
import os as os
import operator as op


def colourbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    figle = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return figle.colorbar(mappable, cax=cax, format='%g')


def correlate(epoch_1=None, epoch_2=None, clipped_side=400, clip_only=False):
    from numpy.fft import fft2, ifft2, fftshift
    if clip_only:
        mid_map_x, mid_map_y = epoch_1.shape[1] // 2, epoch_1.shape[0] // 2
        clipped_epoch = epoch_1[mid_map_y - clipped_side // 2:mid_map_y + clipped_side // 2,
                                mid_map_x - clipped_side // 2:mid_map_x + clipped_side // 2
                                ]
        return clipped_epoch

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
        return xcorr


length = 200

root = '/home/broughtonco/documents/nrc/data/IC348/'
files = os.listdir(root)

FirstEpoch = fits.open('/home/broughtonco/documents/nrc/data/IC348/IC348_20151222_00019_850_EA3_cal.fit')
FirstEpochDate = FirstEpoch[0].header['UTDATE']
FirstEpochData = FirstEpoch[0].data[0]
FED_MidMapX, FED_MidMapY = FirstEpochData.shape[1] // 2, FirstEpochData.shape[0] // 2
FirstEpochData = FirstEpochData[FED_MidMapY-length:FED_MidMapY+length, FED_MidMapX-length:FED_MidMapX+length]
DataSetsPSD, DataSetsAC, radiuses = [], [], []

Dat_Titles = ['Map', 'PowerSpectrum', 'AutoCorr', 'XCorr']
dictionary1 = {}

# DataSets
for fn in files:
    if os.path.isfile(root+'/'+fn) and fn[-4:] != '.txt':
        hdul = fits.open(root+'/'+fn)
        date = hdul[0].header['UTDATE']
        obj = hdul[0].header['OBJECT']
        dictionary1[date] = hdul[0]

for item in sorted(dictionary1.items(), key=op.itemgetter(0)):
    G2D = item[1].data[0]
    date = item[1].header['UTDATE']
    obj = item[1].header['OBJECT']

    G2D = correlate(G2D, clip_only=True)
    XCorr = correlate(epoch_1=G2D, epoch_2=FirstEpochData).real
    PSD = fftshift(ifft2(correlate(G2D))).real
    AC = fftshift(correlate(G2D)).real

    Clipped_G2D_LENGTH = np.arange(0, G2D.shape[0])
    loc = list(product(Clipped_G2D_LENGTH, Clipped_G2D_LENGTH))
    MidMapX = AC.shape[1] // 2
    MidMapY = AC.shape[0] // 2
    radius, PSD_pows, AC_pows = [], [], []

    # writing to a fits file
    dat = [G2D.real, PSD.real, AC.real, XCorr.real]

    for data, name in zip(dat, Dat_Titles):
        if os.path.exists('/home/broughtonco/documents/nrc/data/IC348/GeneratedMapsFits/{}_{}_{}.fit'.format(
                obj, date, name)):
            os.remove('/home/broughtonco/documents/nrc/data/IC348/GeneratedMapsFits/{}_{}_{}.fit'.format(
                obj, date, name))
        hdu = fits.PrimaryHDU(data)
        hdu.writeto(
            '/home/broughtonco/documents/nrc/data/IC348/GeneratedMapsFits/{}_{}_{}.fit'.format(obj, date, name))

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
    im1 = MapIM.imshow(G2D, origin='lower', cmap='magma')
    colourbar(im1)

    acIM.set_title('Auto-correlation')
    im2 = acIM.imshow(AC.real, origin='lower', cmap='magma')
    colourbar(im2)

    psdIM.set_title('Power Spectrum')
    im3 = psdIM.imshow(PSD.real, origin='lower', cmap='magma')
    colourbar(im3)

    xcorrIM.set_title('X-Correlation')
    im4 = xcorrIM.imshow(XCorr.real, origin='lower', cmap='magma')
    colourbar(im4)

    # row 3

    AvgScatterPSD.set_title('Dispersion of Signal power at a a given frequency')
    AvgScatterPSD.set_xlabel('Frequency')
    AvgScatterPSD.set_ylabel('Power')
    AvgScatterPSD.set_yscale('log')
    AvgScatterPSD.set_ylim(bottom=10**-7, top=10**7)
    im5 = AvgScatterPSD.scatter(radius, PSD_pows, marker=',', lw=0, alpha=0.3, color='red', s=1)

    AvgScatterAC.set_title('Auto Correlation')
    AvgScatterAC.set_xlabel('place holder')
    AvgScatterAC.set_ylabel('p-h')
    AvgScatterAC.set_yscale('log')
    AvgScatterAC.set_aspect('auto')
    AvgScatterAC.set_ylim(bottom=10**-4, top=10**4)
    im6 = AvgScatterAC.scatter(radius, AC_pows, marker=',', lw=0, alpha=0.3, color='red', s=1)

    fig1.suptitle('{} IC348 epoch'.format(date))

    # plt.show()
    fig1.savefig('/home/broughtonco/documents/nrc/data/IC348/IC348_plots/epochs/{}'.format(date))

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
    _, PSDd = zip(*sorted(list(zip(Radius_Data, PSDDatSet)), key=op.itemgetter(0)))
    NEW_AC_DATA.append(np.array(ACd))
    NEW_PSD_DATA.append(np.array(PSDd))
radius = r
np.save('/home/broughtonco/documents/nrc/data/IC348/Datafiles/IC348_PSD.npy', np.array(NEW_PSD_DATA))
np.save('/home/broughtonco/documents/nrc/data/IC348/Datafiles/IC348_AC.npy', np.array(NEW_AC_DATA))
# noinspection PyUnboundLocalVariable
np.save('/home/broughtonco/documents/nrc/data/IC348/Datafiles/IC348_radius.npy', np.array(radius))
