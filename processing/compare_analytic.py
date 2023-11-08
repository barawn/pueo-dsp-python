import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

impulse_scale = 1000
cw_scale = 1000

def moving_average(a, n=16):
    # calculate moving average as the difference
    # of two points in the cumulative sum
    ret = np.cumsum(a)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n-1:]/n

sw_lp = np.array( [ 0, -23,
                    0, 105,
                    0, -263,
                    0, 526,
                    0, -949,
                    0, 1672,
                    0, -3216,
                    0,
                    10342, 16384, 10342,               
                    0, -3216,
                    0, 1672,
                    0, -949,
                    0, 526,
                    0, -263,
                    0, 105,
                    0, -23 ] )/32768
ht = np.array( [ 0,
                 -2/(7*np.pi), 0,
                 -2/(5*np.pi), 0,
                 -2/(3*np.pi), 0,
                 -2/(1*np.pi), 0,
                  2/(1*np.pi), 0,
                  2/(3*np.pi), 0,
                  2/(5*np.pi), 0,
                  2/(7*np.pi), 0] )

sx = cw_scale*np.sin(np.linspace(0, 40*np.pi, 201))
sig = np.zeros(201)
sig[38] = impulse_scale
sf = signal.lfilter( sw_lp, [1], sig )
plt.plot(sf)
plt.show()
spsx = sx + sf
plt.plot(spsx)
plt.show()
fspsx = signal.lfilter(ht, [1], spsx)
# compensate for delay
fspsx = fspsx[8:]
# trim last samples in signal
spsx = spsx[0:len(spsx)-8]
plt.plot(fspsx)
plt.plot(spsx)
plt.show()

# calculate simple square
sq = spsx*spsx
# calculate analytic magnitude
mag = spsx*spsx + fspsx*fspsx
plt.plot(sq)
plt.plot(mag)
plt.show()
# decimate by 2
mag = mag[::2]
# moving average by 16 (2 clocks, 8 samp/clock), computed every clock (8 smp)
mov_sq = moving_average(sq, n=16)[::8]
# moving average by 8 (2 clocks, 4 samp/clock), computed every clock (4 smp)
mov_mag = moving_average(mag, n=8)[::4]

plt.plot(mov_sq)
plt.plot(mov_mag)
plt.show()
