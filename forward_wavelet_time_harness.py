""""harness for computing forward frequency domain wavelet transform, take input .dat file in frequency domain (columns frequency, real part(h(f)), imag part(h(f))"
write to .dat file in wavelet domain (Nt rows by Nf columns)"""

import sys
from time import perf_counter
import numpy as np

from transform_time_funcs import transform_wavelet_time

if __name__=='__main__':
    if len(sys.argv)!=7:
        print("transform_time.py filename_in filename_freq_out dt Nt Nf mult")
        sys.exit(1)

    #transform parameters
    file_in = sys.argv[1]
    file_out = sys.argv[2]

    dt = np.float64(sys.argv[3])
    Nt = np.int64(sys.argv[4])
    Nf = np.int64(sys.argv[5])
    mult = np.int64(sys.argv[5])

    print('begin loading data file')
    t0 = perf_counter()
    #the frequency domain representation
    ts_in,signal_time = np.loadtxt(file_in).T
    t1 = perf_counter()
    print('loaded input file in %5.3fs'%(t1-t0))

    ND = Nt*Nf
    Tobs = dt*ND

    #time and frequency grids
    ts = dt*np.arange(0,ND)
    assert np.all(ts==ts_in)

    t0 = perf_counter()
    wave_freq = transform_wavelet_time(signal_time,Nf,Nt,ts)
    t1 = perf_counter()

    print('got time domain transform in %5.3fs'%(t1-t0))

    t4 = perf_counter()
    np.savetxt(file_out,wave_freq)
    t5 = perf_counter()
    print('saved file in %5.3fs'%(t5-t4))
