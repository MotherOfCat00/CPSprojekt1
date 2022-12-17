import array
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fft import fft,  fftfreq
from scipy import signal as sig
from scipy.signal import find_peaks
import sys
np.set_printoptions(threshold=sys.maxsize)

#  filepath ex. D:\hiperkapnia\dta\e[3]_[h]_[10].csv
SAMPLE_RATE = 200  # Hertz
T = 1/SAMPLE_RATE
path_ = r"FILEPATH"
t0 = 0.000
q = 1


def open_file(path):
    filepath = path
    file = pd.read_csv(filepath).rename(columns={"fv_r__[fv_r__]": "FVR", "fv_l__[fv_l__]": "FVL", "fv_r[fv_r]": "FVR",
                                                 "fv_l[fv_l]": "FVL"})
    fvl = file['FVL'].to_numpy()
    fvr = file['FVR'].to_numpy()
    dateTime = file["DateTime"].to_numpy()
    return fvl, fvr, dateTime


# Function to create time vector
def Date(DateTime, t0, T):
    time = array.array('f', [0])
    for i in range(len(DateTime) - 1):
        t0 = t0 + T
        time.append(t0)
    return time


# Downsample signal
def downsample(signal):
    global q
    q = 4
    f_new = SAMPLE_RATE/q
    signal = sig.decimate(signal, q)
    return signal, f_new, q


# Find amplitude of signal
def amplitude(signal):
    peaks, properties = find_peaks(signal, prominence=10, width=4)
    amp_arr = []
    y = 0
    for i in range(len(peaks)):
        peak_index = peaks[i]
        fun_minimum = np.min(signal[y:peak_index])
        y = peaks[i]
        amplitude = signal[peak_index] - fun_minimum
        amp_arr.append(amplitude)
    mean_AMP = np.mean(amp_arr)
    return mean_AMP


#  Filtering
def iirfilter(signal, q):
    fs = SAMPLE_RATE/q
    wo = [0.5, 8]
    b, a = sig.iirfilter(4, Wn=wo, fs=fs, btype="bandpass", ftype="butter")
    y_lfilter = sig.filtfilt(b, a, signal)
    return y_lfilter


# FFT
def fourier_transform(signal, q):
    N = len(signal)
    fs = SAMPLE_RATE/q
    T = 1/fs
    hanning = sig.windows.hamming(N)

    FFT = fft(signal*hanning)
    FFT_freq = fftfreq(N, T)[:N // 2]
    freq_filter = np.where((FFT_freq >= 0.66) & (FFT_freq <= 3.0))
    FFT_freq = FFT_freq[freq_filter]

    FFT_range = FFT[freq_filter]
    corr = 2 / N
    peaks_ind, properties = find_peaks(np.abs(FFT_range * corr), height=1.0, width=1, distance=20)

    peaks_high = properties["peak_heights"]
    result = np.where(peaks_high == np.amax(peaks_high))
    peaks_frequency = peaks_ind[result[0]]
    max_freq = FFT_freq[peaks_frequency]
    AMP_FFT = np.abs(FFT_range[peaks_frequency])
    AMP_FFT = AMP_FFT*4/N
    return N, FFT_freq, FFT_range, AMP_FFT


FVL, FVR, DateTime = open_file(path_)
sign = FVL
sign = sign
sign = sig.detrend(sign)

time = Date(DateTime, t0, T)

sig_decimate, f_new, q_value = downsample(sign)
time_vector = Date(sig_decimate, t0, 1 / f_new)

filtered = iirfilter(sign, q=1)
filtered_decimate = iirfilter(sig_decimate, q_value)

N2, fft_freq, FFT, fft_AMP = fourier_transform(filtered, q=1)
N3, fft_freq_ds, FFT_ds, FFT_AMP_ds = fourier_transform(filtered_decimate, q_value)

raw_amplitude = amplitude(sign)  # amplitude of raw signal
ds_amplitude = amplitude(sig_decimate)  # amplitude of downsampled signal
filt_amplitude = amplitude(filtered)  # amplitude of filtered signal
filt_ds_amplitude = amplitude(filtered_decimate)  # amplitude filtered (downsampled) signal

plt.figure()
plt.plot("Raw signal", raw_amplitude, 'y*', markersize=14)
plt.plot("Downsampled", ds_amplitude, 'y*', markersize=14)
plt.plot("Filtered", filt_amplitude, 'y*', markersize=14)
plt.plot("Filtered (downsampled)", filt_ds_amplitude, 'y*', markersize=14)
plt.ylabel("AMP_CBFV")

plt.figure(figsize=[6.4, 2.4])
plt.plot(time, sign, label="Raw signal")
plt.plot(time_vector, sig_decimate, label="Decimate")
plt.plot(time, filtered, alpha=0.8, lw=3, label="Filtered")
plt.plot(time_vector, filtered_decimate, label="Filtered (decimated)")
plt.xlabel("Time / s")
plt.ylabel("CBFV")
plt.legend(loc="lower center", bbox_to_anchor=[0.5, 1], ncol=2, fontsize="smaller")
plt.xlim([0, 60])

plt.figure()
plt.axes(xlabel="freq", ylabel="...")
plt.title("FFT")
plt.vlines(fft_freq, ymin=0, ymax=2.0/N2 * np.abs(FFT), colors='b')
plt.vlines(fft_freq_ds, ymin=0, ymax=2.0/N3 * np.abs(FFT_ds), colors='r')
plt.grid()
plt.show()
