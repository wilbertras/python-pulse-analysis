import numpy as np
import matplotlib.pyplot as plt
import functions as f
from scipy.signal import fftconvolve
from scipy.fft import fft, ifft

import matplotlib as mpl


try:
    plt.style.use('matplotlibrc')
except:
    print('mpl stylesheet not used')
    pass

fig, ax = plt.subplots()
window = f.get_window('exp', 300)
len_onesided = int(np.round(len(window)/2)+1)
ax.plot(window)

fig, ax = plt.subplots()
wfft = fft(window)
ax.plot(wfft[1:])

fig, ax = plt.subplots()
ax.plot(ifft(np.hstack((wfft[0], wfft[1:len_onesided], wfft[len_onesided:]))))
ax.plot(ifft(wfft), c='r', ls='--')
plt.show()

