import numpy as np
from scipy.io import wavfile

########################## 1. Read in the audio file ##########################

sample_rate, signal = wavfile.read('sample.wav')
print('Sample rate: {} sammples per second'.format(sample_rate))


################ 2. Pre-emphasis (Normalize signal amplitudes) ################

emphasized_signal = np.append(signal[0], signal[1:] - 0.97*signal[:-1])
signal_length = len(emphasized_signal)


############### 3. Framing (Split the signal into short frames) ###############

# we want to set frame length = 25 ms & stride = 10 ms
frame_length = int(round(0.025 * sample_rate))  # length in number of samples
stride = int(round(0.01 * sample_rate))
n_frames = int(np.ceil((signal_length - frame_length) / stride))
print()
print("Number of frames: ", n_frames)

indices = np.tile(np.arange(frame_length), (n_frames, 1)) + \
          np.tile(np.arange(0, stride*n_frames, stride), (frame_length, 1)).T

frames = emphasized_signal[indices]  # shape: (n_frames, frame_length)


################### 4. Apply the window (hamming window) ######################

frames *= np.hamming(frame_length)


########## 5. Perform Fast Fourier Transform & Obtain Power Spectrum ##########

nfft = 512  # performing 512-point Discrete Fourier Transform
# magnitudes of FFT:
fft_magnitudes = np.abs(np.fft.rfft(frames, nfft))
power_spectrum = (fft_magnitudes**2) / nfft


############### 6. Apply Mel Filter Bank to the Power Spectrum ################

n_filters = 40  # number of filters in the mel filter bank
min_mel_freq = 0
max_mel_freq = 2595*np.log10(1 + (sample_rate/2)/700)
mel_points = np.linspace(min_mel_freq, max_mel_freq, n_filters+2)
hz_points = 700*(10**(mel_points/2595) - 1)
fft_bins = np.floor(((nfft + 1) * hz_points) / sample_rate)
filter_banks = np.zeros((n_filters, int(nfft/2 + 1)))

for m in range(1, n_filters + 1):
    for k in range(int(fft_bins[m-1]), int(fft_bins[m])):
        filter_banks[m-1, k] = (k - fft_bins[m-1]) / (fft_bins[m] - fft_bins[m-1])
        
    for k in range(int(fft_bins[m]), int(fft_bins[m+1])):
        filter_banks[m-1, k] = (fft_bins[m+1] - k) / (fft_bins[m+1] - fft_bins[m])

filter_outputs = np.dot(power_spectrum, filter_banks.T)  # shape: (n_frames, n_filters)
filter_outpus = np.where(filter_outputs == 0,
                         np.finfo(float).eps,
                         filter_outputs)   # adjustment for numerical stability
filter_outputs_db = 20 * np.log10(filter_outputs)


######################## 7. Extract 12 MFCC features ##########################
############### by performing Discrete Cosine Transform (dct) #################

from scipy.fftpack import dct

n_mfcc_features = 12
mfcc = dct(filter_outputs_db, 
           type = 2,
           axis = 1,
           norm = "ortho")[:, 1: n_mfcc_features + 1]  # shape: (n_frames, 12)


############################ 8. Mean Normalization ############################

mfcc -= np.mean(mfcc, axis = 0) + 1e-8
print()
print("Shape of MFCC matrix: ", mfcc.shape)  # (n_frames, 12)
print()
