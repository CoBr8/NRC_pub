import numpy as np
import matplotlib.pyplot as plt

args = \
    np.loadtxt('/home/broughtonco/documents/nrc/data/OMC23/Datafiles/OMC23_M_B_table.table', unpack=True)
x = args[0]

fig = plt.figure(figsize=(10, 10))
grid = plt.GridSpec(nrows=4, ncols=4, hspace=0.6, wspace=0.4)

hdr = 'Epoch, Julian Date, Elevation, Tau-225, RMS, Cal_f , Cal_f err, ' \
      'B (5), B Err (5), m / B (5), ' \
      'B (10), B Err (10), m / B (10), ' \
      'B (15), B Err (15), m / B (15) '
hdr = hdr.split(sep=', ')

i=0
for j, y in enumerate(args):
    title = hdr[j]
    j = j % 4
    ax = fig.add_subplot(grid[i, j])
    ax.set_title(title+' vs Epoch')
    if j==3:
        i += 1
    ax.scatter(x, y, marker='.', color='red', lw=0, alpha= .6)

fig.savefig('/home/broughtonco/documents/nrc/data/OMC23/OMC23_plots/OMC23_MetaData.png')
plt.close()
