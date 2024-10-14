from WDMWaveletTransforms.wavelet_transforms import transform_wavelet_freq
from WDMWaveletTransforms.wavelet_transforms import inverse_wavelet_freq

import numpy as np
import matplotlib.pyplot as plt

def test_roundtrip():
    f0 = 1
    Nf = 8
    Nt = 4
    N = Nf*Nt
    dt = 0.1
    freq = np.fft.rfftfreq(N, dt)
    hf = np.zeros_like(freq, dtype=np.complex128)
    f0_idx = np.argmin(np.abs(freq - f0))
    hf[f0_idx] = 1.0

    wavelet = transform_wavelet_freq(hf, Nf=Nf, Nt=Nt)
    freqseries_reconstructed = inverse_wavelet_freq(wavelet, Nf=Nf, Nt=Nt)

    plt.figure()
    plt.plot(
        np.abs(hf), 'o-', label=f"Original {hf.shape}"
    )
    plt.plot(
         np.abs(freqseries_reconstructed), '.', color='tab:red', label=f"Reconstructed {freqseries_reconstructed.shape}"
    )
    plt.legend()
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency [Hz]")

    plt.savefig(f"test_pure_f0_transform.png")

    assert np.allclose(hf.shape, freqseries_reconstructed.shape)
    assert np.allclose(hf, freqseries_reconstructed)


