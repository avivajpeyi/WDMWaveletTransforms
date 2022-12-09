""""test that both inverse functions perform as specified in stored dat files"""
from time import perf_counter
import numpy as np

import pytest

from inverse_wavelet_freq_funcs import inverse_wavelet_freq,inverse_wavelet_freq_time
from inverse_wavelet_time_funcs import inverse_wavelet_time
from transform_time_funcs import transform_wavelet_time
from transform_freq_funcs import transform_wavelet_freq,transform_wavelet_freq_time
from fft_funcs import irfft,rfft

if __name__=='__main__':
    #transform parameters
    dt = 5.
    Nt = 8192
    Nf = 2048
    mult = 16

    ND = Nt*Nf
    Tobs = dt*ND

    #time and frequency grids
    ts = np.arange(0,ND)*dt
    fs = np.arange(0,ND//2+1)*1/(Tobs)
    print("dt=",dt,"Nt=",Nt,"Nf=",Nf,"mult=",mult)

    #get a wavelet representation of a signal
    print('begin loading data files')
    t0 = perf_counter()

    #the initial data
    signal_time = np.random.normal(0.,1.,ND)
    signal_freq = rfft(signal_time)
    wave_in = transform_wavelet_freq(signal_freq,Nf,Nt,fs)

    t1 = perf_counter()

    print('generated data in             %10.7fs'%(t1-t0))

    rfft(signal_time)

    n_run = 10
    t6 = perf_counter()
    for itrm in range(n_run):
        rfft(signal_time)
    t7 = perf_counter()

    time_scale = (t7-t6)/n_run

    print('got time->freq in             %10.7fs %12.7f'%((t7-t6)/n_run,(t7-t6)/n_run/time_scale))

    inverse_wavelet_freq(wave_in,Nf,Nt,dt)

    irfft(signal_freq)

    n_run = 10
    t2 = perf_counter()
    for itrm in range(n_run):
        irfft(signal_freq)
    t3 = perf_counter()

    print('got freq->time in             %10.7fs %12.7f'%((t3-t2)/n_run,(t3-t2)/n_run/time_scale))

    n_run = 10
    t0 = perf_counter()
    for itrm in range(n_run):
        inverse_wavelet_freq(wave_in,Nf,Nt,dt)
    t1 = perf_counter()

    print('got wavelet->freq in          %10.7fs %12.7f'%((t1-t0)/n_run,(t1-t0)/n_run/time_scale))

    inverse_wavelet_time(wave_in,Nf,Nt,dt,mult=mult)

    n_run = 10
    t4 = perf_counter()
    for itrm in range(n_run):
        inverse_wavelet_time(wave_in,Nf,Nt,dt,mult=mult)
    t5 = perf_counter()

    print('got wavelet->time in          %10.7fs %12.7f'%((t5-t4)/n_run,(t5-t4)/n_run/time_scale))

    inverse_wavelet_freq_time(wave_in,Nf,Nt,dt)

    n_run = 10
    t8 = perf_counter()
    for itrm in range(n_run):
        inverse_wavelet_freq_time(wave_in,Nf,Nt,dt)
    t9 = perf_counter()

    print('got wavelet->freq->time in    %10.7fs %12.7f'%((t9-t8)/n_run,(t9-t8)/n_run/time_scale))

    transform_wavelet_freq(signal_freq,Nf,Nt,fs)

    n_run = 10
    t10 = perf_counter()
    for itrm in range(n_run):
        transform_wavelet_freq(signal_freq,Nf,Nt,fs)
    t11 = perf_counter()

    print('got freq->wavelet in          %10.7fs %12.7f'%((t11-t10)/n_run,(t11-t10)/n_run/time_scale))

    transform_wavelet_time(signal_time,Nf,Nt,ts,mult=mult)

    n_run = 10
    t12 = perf_counter()
    for itrm in range(n_run):
        transform_wavelet_time(signal_time,Nf,Nt,ts,mult=mult)
    t13 = perf_counter()

    print('got time->wavelet in          %10.7fs %12.7f'%((t13-t12)/n_run,(t13-t12)/n_run/time_scale))

    transform_wavelet_freq_time(signal_time,Nf,Nt,ts)

    n_run = 10
    t14 = perf_counter()
    for itrm in range(n_run):
        transform_wavelet_freq_time(signal_time,Nf,Nt,ts)
    t15 = perf_counter()

    print('got time->freq->wavelet in    %10.7fs %12.7f'%((t15-t14)/n_run,(t15-t14)/n_run/time_scale))
