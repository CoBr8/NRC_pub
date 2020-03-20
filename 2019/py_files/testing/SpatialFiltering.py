import numpy as np
from numpy import pi
from numpy.fft import fft2, ifft2
from astropy.io import fits
from astropy.modeling.models import Gaussian2D
import matplotlib.pyplot as plt


def colourbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


# basic definitions
NoiPer = 0.05
ClipVal = 400
deg2rad = 180 / pi
AmplitudeMultiplier = 1

MAP_PER = 2 / 100

# Importing image for comparison
hdulOne = fits.open('/home/broughtonco/documents/nrc/data/IC348/IC348_20151222_00019_850_EA3_cal.fit')
MapOnePrimary = hdulOne['primary']
MapOne = MapOnePrimary.data[0]
MIDPOINTOne = np.array([MapOne.shape[1] - 1, MapOne.shape[0] - 1]) // 2
MapOneClipped = MapOne[
                MIDPOINTOne[1] - ClipVal // 2: MIDPOINTOne[1] + ClipVal // 2,
                MIDPOINTOne[0] - ClipVal // 2: MIDPOINTOne[0] + ClipVal // 2
                ]
base = np.zeros(shape=(
    MapOneClipped.shape[1] * 2,
    MapOneClipped.shape[0] * 2))
PadOneMap = np.copy(base)
PadOneMap[
    base.shape[1] // 4: 3 * base.shape[1] // 4,
    base.shape[0] // 4: 3 * base.shape[0] // 4
    ] += MapOneClipped

FFTOne = fft2(PadOneMap)
PowSpecOne = FFTOne * FFTOne.conj()
PowSpecOneRoll = np.roll(PowSpecOne.real, (PowSpecOne.shape[0] - 1) // 2, axis=(0, 1))

# +---------------------+
# | end of the real map |
# +---------------------+

# generating "ideal" map
len_x, len_y = 800, 800
x = np.arange(len_x)
y = np.arange(len_y)
X, Y = np.meshgrid(x, y)

GaussInParams1 = [2.3 * AmplitudeMultiplier, len_x // 2 - 1, len_y // 2 - 1, 2, 2, 0]
G2D = Gaussian2D().evaluate(X, Y, *GaussInParams1)
GaussInParams2 = [1 * AmplitudeMultiplier, len_x // 2 - 201, len_y // 2 - 31, 4, 5, 30]
G2D2 = Gaussian2D().evaluate(X, Y, *GaussInParams2)
GaussInParams3 = [3 * AmplitudeMultiplier, len_x // 2 - 1 + 5, len_y // 2 - 51, 4, 4, 0]
G2D3 = Gaussian2D().evaluate(X, Y, *GaussInParams3)

noi = G2D.max() * NoiPer * np.random.normal(size=G2D.shape)

StructParams1 = [0.5 * AmplitudeMultiplier, 382, 398, 5, 2, 30 * deg2rad]
Struct1 = Gaussian2D().evaluate(X, Y, *StructParams1)
StructParams2 = [0.5 * AmplitudeMultiplier, 377, 386, 8, 3, 95 * deg2rad]
Struct2 = Gaussian2D().evaluate(X, Y, *StructParams2)
StructParams3 = [0.5 * AmplitudeMultiplier, 388, 368, 7, 2, 125 * deg2rad]
Struct3 = Gaussian2D().evaluate(X, Y, *StructParams3)

G2D += G2D2 + G2D3 + Struct1 + Struct2 + Struct3 + noi

FFT_G2D = fft2(G2D)
FFT_G2D: np.ndarray = np.roll(FFT_G2D, (FFT_G2D.shape[0] // 2) - 1, axis=(0, 1))

MidMapX = FFT_G2D.shape[1] // 2 - 1
MidMapY = FFT_G2D.shape[0] // 2 - 1

for i in np.arange(FFT_G2D.shape[1]):
    for j in np.arange(FFT_G2D.shape[0]):
        if ((i - MidMapX) ** 2 + (j - MidMapY) ** 2) ** (1 / 2) <= FFT_G2D.shape[0]*MAP_PER:
            FFT_G2D[i, j] = 0

FFT_G2D: np.ndarray = np.roll(FFT_G2D, (FFT_G2D.shape[0] - 1) // 2, axis=(0, 1))
FFT_G2D_CONJ = FFT_G2D.conj()
PSD = FFT_G2D * FFT_G2D_CONJ
AC: np.ndarray = ifft2(PSD)

# plotting
fig1, ((MAPim, FFTim, ACim, PSDim),
       (MAPdat, FFTdat, ACdat, PSDdat)) = plt.subplots(2, 4, figsize=(24, 12))

# single calculation to improve speed
MAP = ifft2(FFT_G2D).real
FFTMAP = np.roll(FFT_G2D, (FFT_G2D.shape[0] - 1) // 2, axis=(0, 1)).real
ACMAP = np.roll(AC.real, (AC.real.shape[0] - 1) // 2, axis=(0, 1))
PSDMAP = np.roll(PSD.real, (PSD.shape[0] - 1) // 2, axis=(0, 1))

# row 1
MAPim.set_title('Map')
MAPim.set_axis_off()
MAPim = MAPim.imshow(MAP[MAP.shape[0] // 2:], origin='lower', cmap='magma')
colourbar(MAPim)

FFTim.set_title('Fast Fourier Transform')
FFTim.set_axis_off()
FFTim = FFTim.imshow(FFTMAP, origin='lower', cmap='magma')
colourbar(FFTim)

ACim.set_title('Auto-correlation')
ACim.set_axis_off()
ACim = ACim.imshow(ACMAP, origin='lower', cmap='magma')
colourbar(ACim)

PSDim.set_title('Power Spectral Density')
PSDim.set_axis_off()
PSDim = PSDim.imshow(PSDMAP, origin='lower', cmap='magma')
colourbar(PSDim)

# row 2
MAPdat.set_title('IC348 Map')
MAPdat.set_axis_off()
MAPdat = MAPdat.imshow(MapOne.real, origin='lower', cmap='magma')
colourbar(MAPdat)

FFTdat.set_title('FFT')
FFTdat.set_axis_off()
FFTdat = FFTdat.imshow(np.roll(FFTOne.real, (FFTOne.shape[0] - 1) // 2, axis=(0, 1)), origin='lower', cmap='magma')
colourbar(FFTdat)

ACdat.set_title('Auto-Correlation')
ACdat.set_axis_off()
ACdat = ACdat.imshow(np.roll(ifft2(PowSpecOne),
                     (PowSpecOne.shape[0] - 1) // 2,
                     axis=(0, 1)).real,
                     origin='lower',
                     cmap='magma'
                     )
colourbar(ACdat)

PSDdat.set_title('Power Spectral Density')
PSDdat.set_axis_off()
PSDdat = PSDdat.imshow(PowSpecOneRoll.real, origin='lower', cmap='magma')
colourbar(PSDdat)

plt.show()
