import numpy as np
from scipy.io import wavfile
from spectrogram_generator import spectrogram_from_file


########################## 1. Read in the audio file ##########################

sample_rate, signal = wavfile.read('sample.wav')
print('Sample rate: {} sammples per second'.format(sample_rate))


################ 2. Pre-emphasis (Normalize signal amplitudes) ################

emphasized_signal = np.append(signal[0], signal[1:] - 0.97*signal[:-1])
signal_length = len(emphasized_signal)
print("Length of Audio: {} samples ".format(signal_length))
print()


############### 3. Framing (Split the signal into short frames) ###############

# we want to set frame length = 25 ms & stride = 10 ms
fl_in_seconds = 0.025
fl_in_samples = int(round(fl_in_seconds * sample_rate))  # length in number of samples

stride_in_seconds = 0.01
stride_in_samples = int(round(stride_in_seconds * sample_rate))
n_frames = int(np.ceil((signal_length - fl_in_samples) / stride_in_samples))

print("Frame Length: {} samples ({} ms)".format(fl_in_samples, fl_in_seconds*1000))
print("Stride Length: {} samples ({} ms)".format(stride_in_samples, stride_in_seconds*1000))
print("Resulting number of frames: {} frames".format(n_frames))
print()

indices = np.tile(np.arange(fl_in_samples), (n_frames, 1)) + \
          np.tile(np.arange(0, stride_in_samples * n_frames, stride_in_samples), (fl_in_samples, 1)).T

frames = emphasized_signal[indices]  # shape: (n_frames, fl_in_samples)


################### 4. Apply the window (hamming window) ######################

frames *= np.hamming(fl_in_samples)


########## 5. Perform Fast Fourier Transform & Obtain Power Spectrum ##########

n_fft = 512  # performing 512-point Discrete Fourier Transform
# magnitudes of FFT:
fft_magnitudes = np.abs(np.fft.rfft(frames, n_fft))
power_spectrum = (fft_magnitudes**2) / n_fft


############### 6. Apply Mel Filter Bank to the Power Spectrum ################

n_filters = 40  # number of filters in the mel filter bank
min_mel_freq = 0
max_mel_freq = 2595*np.log10(1 + (sample_rate/2)/700)
mel_points = np.linspace(min_mel_freq, max_mel_freq, n_filters+2)
hz_points = 700*(10**(mel_points/2595) - 1)
fft_bins = np.floor(((n_fft + 1) * hz_points) / sample_rate)
filter_banks = np.zeros((n_filters, int(n_fft/2 + 1)))

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
print("Shape of Mel Filter Bank:    ", filter_outputs_db.shape)


######################## 7. Extract 12 MFCC features ##########################
############### by performing Discrete Cosine Transform (dct) #################

from scipy.fftpack import dct

n_mfcc_features = 12
mfcc = dct(filter_outputs_db, 
           type = 2,
           axis = 1,
           norm = "ortho")[:, 1: n_mfcc_features + 1]


############################ 8. Mean Normalization ############################

mfcc -= np.mean(mfcc, axis = 0) + 1e-8
print("Shape of MFCC matrix:        ", mfcc.shape)  


########################### 9. Spectrogram Features ###########################

spec_matrix = spectrogram_from_file("sample.wav",
                                    frame_stride = 10,
                                    frame_window = 20,
                                    max_freq = 4000)

print("Shape of Spectrogram matrix: ", spec_matrix.shape)
print()
print("Now, either the MFCC matrix or Spectrogram matrix can be fed in as input to an audio machine learning tasks.")
print()