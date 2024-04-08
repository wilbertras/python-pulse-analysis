import numpy as np
from scipy.fft import fft
from scipy.signal import welch
import matplotlib.pyplot as plt


def pulse_model(pulses, pw):
    """
    pulse_model computes  that serves as input for the optimal filter.
    :param pulses:              numpy.ndarray, MxN array of M pulses of N length
    :param pw:                  int, pulse window length
    :return norm_pulse_fft:     numpy.ndarray, the onesided fft of the normalised mean pulse
    """
    len_onesided = round(pw / 2) + 1

    mean_pulse = np.mean(pulses, axis=0)
    max_pulse = np.amax(mean_pulse)
    norm_pulse = mean_pulse / max_pulse
    
    norm_pulse_fft = fft(norm_pulse)[:len_onesided]
    return norm_pulse_fft


def noise_model(noise, pw, sf, threshold=None, nr_req_segments=1000):
    """
    noise_model computes the power spectral density of the noise that serves as input for optimal filter. 
    :param noise:           numpy.ndarray, 1D array of noise data
    :param pw:              int, pulse window length
    :param sf:              int, sample frequency of noise data
    :param threshold:       float/bool/int, threshold that is used to detect if there are pulses present in the noise data. The segments with data above the threshold are discarded 
    :param nr_req_segmets:  int, number of desired noise segments to average for construct the average noise PSD
    :return noise_psd:      numpy.ndarray, 1D array of the averaged onesided PSD of the noise
    :return freqs:          numpy.ndarray, 1D array of the frequency data corresponding to the noise_psd
    """
    noise_len = len(noise)
    len_onesided = round(pw / 2) + 1
    noise_psd = np.zeros(len_onesided)

    if not threshold:                           # if there is no threshold give, set the threshold as 5*sigma of the noise data
        std = np.std(noise)
        threshold = np.round(5 * std, decimals=2)
    
    nr_good_segments = 0
    start = 0
    count = 0
    while nr_good_segments < nr_req_segments:           
        start = count * pw
        stop = start + pw
        if stop >= noise_len:  # ensure that the next segment does not run out of the available data 
            print('     WARNING: only %d/%d noise segments obtained with max_bw=%d' % (nr_good_segments, nr_req_segments, pw))
            break
        next_segment = noise[start:stop]
        nr_outliers = np.sum(next_segment > threshold)
        if nr_outliers > 0:  # check whether there are pulses in the data
            count += 1
        else:
            freqs, noise_psd_segment = welch(next_segment, fs=int(sf), window='hamming', nperseg=pw, noverlap=None, nfft=None, return_onesided=True)
            noise_psd += noise_psd_segment  # cumulatively add the PSDs of all the noise segments
            nr_good_segments += 1
            count += 1   
    if nr_good_segments == 0:
        raise Exception('No good noise segments found')
    noise_psd /= nr_good_segments  # compute the avarage PSD
    return noise_psd, freqs


def prefactor(norm_pulse_fft, noise_psd, exclude_dc=True):      
    """
    prefactor computes the normalisation factor for the optimal filter
    :param norm_pulse_fft:  numpy.ndarray, 1D array of pulse model
    :param noise_psd:       numpy.ndarray, 1D array of noise modl
    :param exclude_dc:      bool, whether to exclude the DC value of the fft
    :return factor:         float, normalisation factor

    """
    factor = np.sum(np.absolute(norm_pulse_fft[exclude_dc:])**2/noise_psd[exclude_dc:])
    return factor


def mux_filter(pulse, len_onesided, norm_pulse_fft, noise_psd, norm_factor, exclude_dc=True):
    """
    mux_filter is the function that should resemple which computations need to be done by the MUX setup
    :param pulse:           numpy.ndarray, 1D array of a single pulse
    :param len_onesided:    int, pulse window length
    :param norm_pulse_fft:  numpy.ndarray, pulse model
    :param noise_psd:       numpy.ndarray, noise model
    :norm_factor:           float, normalisation factor
    :param exclude_dc:      bool, whether to exclude the DC value of the fft
    :return H:              float, optimal pulse height
    """
    pulse_fft = fft(pulse)[:len_onesided]
    dH = (norm_pulse_fft.conj() * pulse_fft / noise_psd)
    H = np.real(np.sum(dH[exclude_dc:]) / norm_factor)
    return H




