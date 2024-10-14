"""

Toy model of sine-wave signal and a constant PSD

"""
import matplotlib.pyplot as plt
import numpy as np
from WDMWaveletTransforms.wavelet_transforms import transform_wavelet_time, inverse_wavelet_time
import pytest
import scipy

def test_wavelet_timedomain_snr(f0=20,T=1000, A=1e-3, PSD_AMP=1e-2, Nf=16):
    ########################################
    # Part1: Analytical SNR calculation
    ########################################
    dt = 0.5 / (2 * f0)  # Shannon's sampling theorem, set dt < 1/2*highest_freq
    t = np.arange(0, T, dt)  # Time array
    # round len(t) to the nearest power of 2
    t = t[:2 ** int(np.log2(len(t)))]
    T = len(t) * dt

    y = A * np.sin(2 * np.pi * f0 * t)  # Signal waveform we wish to test

    freq = np.fft.fftshift(np.fft.fftfreq(len(y), dt))  # Frequencies
    df = abs(freq[1] - freq[0])  # Sample spacing in frequency

    y_fft = dt * np.fft.fftshift(np.fft.fft(y))  # continuous time fourier transform [seconds]
    N_f = len(y_fft)
    N_t = len(y)
    PSD = PSD_AMP * np.ones(len(freq))

    # Compute the SNRs
    SNR2_f = 2 * np.sum(abs(y_fft) ** 2 / PSD) * df
    SNR2_t = 2 * dt * np.sum(abs(y) ** 2 / PSD)
    SNR2_t_analytical = (A ** 2) * T / PSD[0]

    ########################################
    # Part2: Wavelet domain
    ########################################

    ND = len(y)
    Nt = ND // Nf
    ND = Nf * Nt

    signal_wavelet = transform_wavelet_time(y, Nf=Nf, Nt=Nt) * np.sqrt(2) * dt

    delta_t = T / Nt
    delta_f = 1 / (2 * delta_t)
    freq_grid = np.arange(0, Nf) * delta_f
    time_grid = np.arange(0, Nt) * delta_t

    psd_wavelet = PSD_AMP * np.ones((Nt, Nf)) * dt

    wavelet_snr2 = np.sum((signal_wavelet * signal_wavelet / psd_wavelet))
    mse = np.mean((y - inverse_wavelet_time(signal_wavelet, Nf=Nf, Nt=Nt)) ** 2)
    print('---------')
    print(f"SNR squared in the frequency domain is = {SNR2_f:.2f}")
    print(f"SNR squared in the time domain (Parseval's theorem) is = {SNR2_t:.2f}", )
    print(f"(pen and paper) Analytical result would predict SNR squared = {SNR2_t_analytical:.2f}")
    print(f"In the wavelet domain, SNR_sqr = {wavelet_snr2:.2f}")
    print(f"Mean squared error in the wavelet domain = {mse:.2f}")
    print('---------')
    assert np.isclose(SNR2_f, wavelet_snr2, atol=1e-2), "SNR in time domain and wavelet domain should be the same"




if __name__ == '__main__':
    test_chirp_signal()
