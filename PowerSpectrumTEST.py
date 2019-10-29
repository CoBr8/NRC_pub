import numpy as np
from numpy.fft import fft,ifft,fft2,ifft2
import matplotlib.pyplot as plt
import matplotlib
from astropy.io import fits as fits

# +===================+
# | basic definitions |
# +===================+

ClipVal = 400

# +=========================+
# | importing the data sets |
# +=========================+

hdulREF = fits.open('/home/broughtonco/documents/nrc/data/IC348/IC348_20151222_00019_850_EA3_cal_smooth.fit')
hdulSCI = fits.open('/home/broughtonco/documents/nrc/data/IC348/IC348_20191009_00032_850_EA3_cal_smooth.fit')

MapRefPrimary = hdulREF['primary']
MapSciPrimary = hdulSCI['primary']

ObjName = MapRefPrimary.header['OBJECT']
REFDate = MapRefPrimary.header['UTDATE']
SCIDate = MapSciPrimary.header['UTDATE']

ReferenceMap = MapRefPrimary.data[0]
ScienceMap   = MapSciPrimary.data[0]

# +============================================+
# | Generating padded and clipped maps for FFT |
# +============================================+

# doubling size of map to centre ReferenceMap and ScienceMap with zero padding around it

MIDPOINTRef = np.array([ReferenceMap.shape[1] - 1, ReferenceMap.shape[0] - 1]) // 2
MIDPOINTSci = np.array([ScienceMap.shape[1] - 1, ScienceMap.shape[0] - 1]) // 2

ReferenceMapClipped = ReferenceMap[
                                MIDPOINTRef[1] - ClipVal // 2 : MIDPOINTRef[1] + ClipVal // 2,
                                MIDPOINTRef[0] - ClipVal // 2 : MIDPOINTRef[0] + ClipVal // 2
                                ]
ScienceMapClipped = ScienceMap[
                            MIDPOINTSci[1] - ClipVal // 2 : MIDPOINTSci[1] + ClipVal // 2,
                            MIDPOINTSci[0] - ClipVal // 2 : MIDPOINTSci[0] + ClipVal // 2
                            ]

base = np.zeros(shape=(
                    ReferenceMapClipped.shape[1] * 2,
                    ReferenceMapClipped.shape[0] *2
                    )
                    )

PaddedReferenceMap = np.copy(base)
PaddedScienceMap = np.copy(base)

PaddedReferenceMap[
                    base.shape[1] // 4: 3 * base.shape[1] // 4,
                    base.shape[0] // 4: 3 * base.shape[0] // 4
                    ] += ReferenceMapClipped

PaddedScienceMap[
                base.shape[1] // 4: 3 * base.shape[1] // 4,
                base.shape[0] // 4: 3 * base.shape[0] // 4
                ] += ScienceMapClipped


# +=======================================+
# calculating cross-correlation using FFT |
# +=======================================+

FFTReference = fft2(PaddedReferenceMap)
FFTScience   = fft2(PaddedScienceMap)
PowSpecRef   = FFTReference * FFTReference.conj()
PowSpecSci   = FFTScience * FFTScience.conj()

PowSpecRef = np.roll(PowSpecRef.real, (PowSpecRef.shape[0]-1)//2, axis=(0,1))
PowSpecSci = np.roll(PowSpecSci.real, (PowSpecSci.shape[0]-1)//2, axis=(0,1))

# +=================+
# | plotting graphs |
# +=================+

plt1mod = -200
plt2mod = 0
plt3mod = +5


n = 2 
y, x   = np.where(PowSpecRef==PowSpecRef.max())
y1, x1 = np.where(PaddedReferenceMap==PaddedReferenceMap.max() )
pow1, freq1 = matplotlib.mlab.psd(PaddedReferenceMap[int(y1)+plt1mod,PaddedReferenceMap.shape[1] // 4: 3 * PaddedReferenceMap.shape[1] // 4],
        Fs = PaddedReferenceMap.shape[1])

pow2, freq2 = matplotlib.mlab.psd(PaddedReferenceMap[int(y1)+plt2mod,PaddedReferenceMap.shape[1] // 4: 3 * PaddedReferenceMap.shape[1] // 4],
	Fs = PaddedReferenceMap.shape[1])
pow3, freq3 = matplotlib.mlab.psd(PaddedReferenceMap[int(y1)+plt3mod,PaddedReferenceMap.shape[1] // 4: 3 * PaddedReferenceMap.shape[1] // 4],
        Fs = PaddedReferenceMap.shape[1])

fig,((plot1,bar1),(plot2,bar2),(plot3,bar3)) = plt.subplots(3,2) # gen. subplots of 3 rows, 2 columns
# plot 1
plot1.plot(PowSpecRef[int(y)+plt1mod,:])
plot1.set_yscale('log')

bar1.bar(freq1, pow1)
# bar1.set_yscale('log')
bar1.set_xlabel('frequency (Hz)')
bar1.set_ylabel('Power')
# plot 2
plot2.plot(PowSpecRef[int(y)+plt2mod,:])
plot2.set_yscale('log')

bar2.bar(freq2, pow2)
bar2.set_yscale('log')
bar2.set_xlabel('frequency (Hz)')
bar2.set_ylabel('Power')
# plot 3
plot3.plot(PowSpecRef[int(y)+plt3mod,:])
plot3.set_yscale('log')

bar3.bar(freq3, pow3)
bar3.set_yscale('log')
bar3.set_xlabel('frequency (Hz)')
bar3.set_ylabel('Power')

plt.show()
