import glob
import numpy as np
import re
from copy import copy
import pickle
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch, resample, windows, fftconvolve
from scipy.fft import fft, ifft
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


def get_bin_files(dir_path, kid_nr, p_read, type='vis'):
    txt = 'KID' + str(kid_nr) + '_' + str(p_read) + 'dBm__TD' + str(type)
    if type == 'vis':
        info_path = dir_path + '/' + txt + '0' + '*_info.dat'
    else:
        info_path = dir_path + '/' + txt + '*_info.dat'
    bin_path = dir_path + '/' + txt + '*.bin'
    list_bin_files = glob.glob(bin_path)
    info_file = glob.glob(info_path)
    if not list_bin_files:
        raise Exception('Please correct folder path as no files were obtained using path:\n%s' % (bin_path))
    list_bin_files = sorted(list_bin_files, key=lambda s: int(re.findall(r'\d+', s)[-2]))
    return list_bin_files, info_file


def get_info(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    f0 = float(re.findall("\d+\.\d+", lines[2])[0]) 
    Qs = re.findall("\d+\.\d+", lines[3])
    Qs = [float(Q) for Q in Qs]
    [Q, Qc, Qi, S21_min] = Qs
    fs = 1 / float(re.findall("\d+\.\d+", lines[4])[0])
    T = float(re.findall("\d+\.\d+", lines[7])[0])
    info = {'f0' : f0, 'Q' : Q, 'Qc' : Qc, 'Qi' : Qi, 'S21_min' : S21_min, 'fs' : fs, 'T' : T}
    return info


def ensure_type(input_value, preferred_types, orNoneType=False):
    """
    Convert input_value to the preferred data type from the list of preferred_types.
    
    Parameters:
        input_value (any): The input value to be converted.
        preferred_types (list): A list of preferred data types. Can include int, float, str, list, tuple, None.
        
    Returns:
        The input_value converted to the preferred data type, or None if conversion fails.
    """
    if isinstance(preferred_types, type):
        preferred_types = [preferred_types]
    elif preferred_types == None:
        return None
    if input_value == None:
        if orNoneType:
            return input_value
        else:
            raise Exception('Cant convert NoneType to preferred types')
    elif isinstance(input_value, str):
        values = input_value.split()
        if len(values) == 0 or input_value == 'None':
            return None
        elif len(values) == 1:
            pass
        else: 
            try:
                input_value = [int(value) for value in values]
            except:
                input_value = [float(value) for value in values]
    if any(isinstance(input_value, type) for type in preferred_types):
        return input_value

    for data_type in preferred_types:
        try:
            if data_type == int:
                return int(input_value)
            elif data_type == float:
                return float(input_value)
            elif data_type == str:
                return str(input_value)
            elif data_type == list:
                return list(input_value)
            elif data_type == tuple:
                return tuple(input_value)
            elif data_type == np.ndarray:
                return np.array(input_value)
            elif data_type == None:
                return None
            else:
                raise ValueError("Unsupported data type")
        except (ValueError, TypeError):
            continue
    if orNoneType:
        return None
    else:
        print('WARNING: Could not convert data type')


def bin2mat(file_path):
    data = np.fromfile(file_path, dtype='>f8', count=-1)
    data = data.reshape((-1, 2))

    I = data[:, 0]
    Q = data[:, 1]

    # From I and Q data to Radius/Magnitude and Phase
    r = np.sqrt(I**2 + Q**2)
    R = r/np.mean(r) # Normalize radius to 1

    P = np.arctan2(Q, I) 
    P = np.pi - P % (2 * np.pi) # Convert phase to be taken from the negative I axis
    return R, P


def plot_bin(file_path):
    response = bin2mat(file_path)[1]
    time = np.arange(len(response)) * 20e-6
    info = get_info(file_path[:-4]+'_info.dat')
    print(info)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(time, response, lw=.2)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('$\\theta$ [rad]')
    line_len = len(file_path) // 2
    ax.set_title(file_path[:line_len] + '\n' + file_path[line_len:])
    ax.set_xlim(time[0], time[-1])


def concat_vis(file_list, discard=True):
    limit = -0.5 * np.pi
    amp = []
    phase = []
    removed = 0
    saturated_Ts = []
    for i, file in enumerate(file_list):
        r, p = bin2mat(file)
        saturated = p <= limit
        if np.any(saturated):
            saturated_Ts.append(i)
            if discard:
                removed += 1
                append = 0
                fig, ax = plt.subplot_mosaic('ab', figsize=(6, 3), sharex=True, sharey=True, constrained_layout=True)
                ax['a'].set_title('phase')
                ax['a'].plot(p, lw=.5)
                # p[saturated] += 2 * np.pi
                ax['b'].set_title('amplitude')
                ax['b'].plot(r, lw=.5)
            else: 
                append = 1
        else:
            append = 1
        if append:
            amp.append(r)
            phase.append(p)
    amp = np.array(amp).flatten()
    phase = np.array(phase).flatten()
    nr_saturated = len(saturated_Ts)
    if nr_saturated:
        print('     WARNING: %d files found with phase <= %.1f pi rad (at T=' % (nr_saturated, limit/np.pi), saturated_Ts, 's)')
    return amp, phase, removed


def smith_coord(P, R):
    '''
    This function returns the phase and amplitude reponse in the Smith chart coordinate systam.
    '''
    # Normalised I and Q
    I_r = -np.cos(P) * R
    Q_r = np.sin(P) * R

    # SMith chart coordinate system
    G = I_r + 1j * Q_r
    z = (1 + G) / (1 - G)

    R_smith = np.real(z)
    X_smith = np.imag(z)
    R_smith -= np.mean(R_smith)
    X_smith -= np.mean(X_smith)
    return R_smith, X_smith


def coord_transformation(response, coord, phase, amp, dark_phase=[], dark_amp=[]):
    dark_too = True
    if len(dark_phase) == 0:
        dark_too = False

    # if dark_too:
    #     dark_phase[dark_phase <= limit] += 2 * np.pi

    if coord == 'smith':
        R, X = smith_coord(phase, amp)
        if dark_too:
            R_dark, X_dark = smith_coord(dark_phase, dark_amp)
        if response == 'R':
            signal = R
            if dark_too:
                dark_signal = R_dark
        elif response == 'X':
            signal = X
            if dark_too:
                dark_signal = X_dark
        else:
            raise Exception('Please input a proper response type ("R" or "X")')
    elif coord == 'circle':
        if response == 'phase':
            signal = phase - np.mean(phase)
            if dark_too:
                dark_signal = dark_phase - np.mean(dark_phase)
        elif response == 'amp':
            signal = (1 - amp) - np.mean(1 - amp)
            if dark_too:
                dark_signal = (1 - dark_amp) - np.mean(1 - dark_amp)
        else:
            raise Exception('Please input a proper response type ("amp" or "phase")')
    else:
        raise Exception('Please input a proper coordinate system ("smith" or "circle")')   
    



    if dark_too:
        return signal, dark_signal
    else:
        return signal


def supersample(signal, num, type='interp', axis=0):
    if type == 'interp':
        l = len(signal)
        x = np.arange(l)  
        x_ss = np.linspace(0, l-1, num)
        return interp1d(x, signal, axis=axis)(x_ss)
    elif type == 'fourier':
        return resample(signal, num, axis=axis)
    else:
        raise Exception('Please input correct supersample type: "interp" or "fourier"')


def peak_model(signal, mph, mpp, pw, sw, align, window, sff, ssf, sstype, buffer, rise_offset, plot_pulse=False):
    '''
    This function finds, filters and aligns the pulses in a timestream data
    '''
    pw *= sff
    sw *= sff
    buffer *= sff
    rise_offset *= sff

    # Smooth timestream data for peak finding
    if sw:    
        smoothed_signal = fftconvolve(signal, window, mode='valid')
    else:
        smoothed_signal = signal

    # Find peaks in data
    locs_smoothed, props_smoothed = find_peaks(smoothed_signal, height=mph, prominence=mpp/2)
    pks_smoothed = props_smoothed['peak_heights']
    nr_pulses = len(locs_smoothed)
    det_locs = copy(locs_smoothed)
    if nr_pulses == 0:
        pulses_aligned = []
        H = []
        sel_locs = []
        filtered_locs = []
        pks_smoothed = []
    else:
        # Assign buffer for pulse alignment. The buffer is applied on both sides of the pulsewindow: buffer + pw + buffer
        buffer_len = int(2*buffer + pw)

        # Filter peak that is too close to the end of data array (too close too start is already filtered in previous step)
        length_signal = len(signal)
        filter_start = locs_smoothed - rise_offset - buffer >= 0
        filter_stop = locs_smoothed + pw + buffer <= length_signal

        # Filter the peaks that are too close to one another
        diff_locs = copy(locs_smoothed)  # make shallow copy of locs
        diff_locs[1:] -= locs_smoothed[:-1]  # compute the spaces in between the peaks
        filter_left = diff_locs >= (buffer + pw)  # find all peaks that are too close to the previous peak
        filter_right = np.hstack((filter_left[1:], True))  # find the peaks that are too close to the following peak, which is just a simple shift of the previous 
        filter = filter_left & filter_right & filter_start & filter_stop  # find all peaks that are far enough from both the previous as the following peak
        locs_smoothed = locs_smoothed[filter] # filter peaks where the space in between is less than pulsewindow + 2*buffer
        pks_smoothed = pks_smoothed[filter]
        nr_far_enough = filter.sum()
        perc_too_close = round(100 * (1 - nr_far_enough / nr_pulses))

        # Cut pulses from timestream and align on peak or rising edge
        sel_locs = copy(locs_smoothed)
        pulses_aligned = []
        idx_align = []
        H = []
        filter_diff = np.ones(len(sel_locs), dtype=bool)
        
        # Setup figure for pluse plotting:
        if plot_pulse:
            fig, axes = plt.subplot_mosaic('abcde;fghij', figsize=(15, 5), constrained_layout=True, sharex=True, sharey=True)
            pos = 'abcdefghij'
            ypos = 'af'
            xpos = 'fghij'
            nr_plots = len(pos)
            plot_count = 0

        for i in range(len(locs_smoothed)):
            loc = locs_smoothed[i]
            left = int(loc - rise_offset - buffer)
            right = int(loc + (pw - rise_offset) + buffer)
            pulse = signal[left:right]    # first take a cut with buffer on both sides based on loc from smoothed data
            smoothed_pulse = smoothed_signal[left:right]
            smoothed_peak = pks_smoothed[i]

            if align == 'peak':
                pulses_aligned.append(pulse[buffer:-buffer])
                idx_align.append(loc)
                full_max = signal[loc]
                H.append(full_max)
                smoothed_loc = rise_offset + buffer
                idx_max = rise_offset + buffer

            elif align == 'edge': 
                # Supersample the peak
                if ssf > 1:
                    # if i == 0:
                    #     pw *= ssf
                    #     buffer *= ssf
                    #     rise_offset *= ssf
                    #     buffer_len *= ssf
                    pulse = supersample(pulse, buffer_len*ssf, type=sstype)
                    smoothed_pulse = supersample(smoothed_pulse, buffer_len*ssf, type=sstype)
                
                smoothed_loc = (buffer + rise_offset)*ssf   # this is a guess of the smoothed peak based on the loc from smoothed data. The true smoothed peak and peak height still have to be determined   
                len_pulse = int(buffer_len*ssf)
                unsmoothed_height = pulse[smoothed_loc]

                if unsmoothed_height <= smoothed_peak:
                    min_height = smoothed_peak
                else:
                    min_height = unsmoothed_height

                # Find non-smoothed peak closest to smoothed peak
                locs_right, props_right = find_peaks(pulse[smoothed_loc-1:], height=min_height, prominence=0) # Find peaks to the right of smoothed peak with at minimal height the value at the smoothed peak
                if len(locs_right) != 0:
                    idx_max_right = smoothed_loc-1 + locs_right[0]
                    idx_max = idx_max_right
                    full_max = props_right['peak_heights'][0]
                else:
                    locs_left, props_left = find_peaks(pulse[:smoothed_loc+1], height=min_height, prominence=0) # Find peaks to the right of smoothed peak with at minimal height the value at the smoothed peak
                    if len(locs_left) != 0:
                        idx_max_left = locs_left[-1]
                        idx_max = idx_max_left
                        full_max = props_left['peak_heights'][0]   
                    else:
                        filter_diff[i] = False
                        continue
                sel_locs[i] = left + int(idx_max/ssf)

                # Align pulses on rising edge   
                half_max = full_max / 2
                rising_edge = idx_max - np.argmax(pulse[-(len_pulse - idx_max)::-1] < half_max) # Find rising edge as the first value closest to half the maximum starting from the peak
                if rising_edge > rise_offset*ssf: # Check
                    shift_start = int(rising_edge - rise_offset*ssf)  # Start cut at rise_offset before rising edge
                    shift_end = int(shift_start + pw*ssf)  # End cut at pulsewindow after rising edge
                    aligned_pulse = pulse[shift_start:shift_end]
                    if len(aligned_pulse) == pw*ssf:
                        pulses_aligned.append(aligned_pulse)
                        idx_align.append(rising_edge)
                        H.append(full_max) 
                    else:
                        filter_diff[i] = False
                else:
                    filter_diff[i] = False
            else:
                raise Exception('Please input a correct aligning method')    
                
            # Option to plot some pulses with their peaks, half maxima and rising edge indicated
            if plot_pulse:
                if plot_count < nr_plots:
                    label = pos[plot_count]
                    ax = axes[label]
                    t = np.linspace(0, pw*ssf, len(pulse))
                    ax.plot(t, pulse, lw=0.5, c='tab:blue', label='pulse', zorder=0)
                    if sw:
                        ax.plot(t, smoothed_pulse, lw=0.5, c='tab:orange', label='smoothed pulse', zorder=1)
                    ax.axhline(mph, lw=0.5, c='tab:red', label='min. peak height', zorder=2)
                    ax.scatter(t[smoothed_loc], smoothed_pulse[smoothed_loc], c='None', edgecolor='tab:orange', marker='v', label='smoothed peak', zorder=3)
                    ax.scatter(t[idx_max], full_max, color='None', edgecolor='tab:green', marker='v', label='peak', zorder=3)
                    if align == 'edge':
                        ax.scatter(t[rising_edge], half_max, color='None', edgecolor='tab:green', marker='s', label='rising edge', zorder=3)
                    if label in xpos:
                        ax.set_xlabel('$\it t$ $[\mu s]$')
                    if label in ypos:
                        ax.set_ylabel('$response$')
                    ax.set_xlim([0, pw])
                    # axes['a'].legend(loc='upper right', ncol=2)
                    axes['a'].legend(bbox_to_anchor=(0., 1.06, 5., .06), loc='upper left',
                    ncols=7, mode="expand", borderaxespad=0., fontsize=9)
                    plot_count += 1          
        pulses_aligned = np.array(pulses_aligned).reshape((-1, pw*ssf))
        idx_align = np.array(idx_align)
        H = np.array(H)
        sel_locs = sel_locs[filter_diff]
        locs_smoothed = locs_smoothed[filter_diff]      
        pks_smoothed = pks_smoothed[filter_diff]
 
        # Print filtered number and percentages
        nr_pulses_aligned = np.shape(pulses_aligned)[0]
        nr_unaligned = filter_diff.sum()
        perc_outliers = round(100 * (1 - nr_unaligned / nr_pulses) - perc_too_close)
        perc_selected = round(100 * nr_pulses_aligned / nr_pulses)
        filtered_locs = np.setdiff1d(det_locs, locs_smoothed)
        print('     N_sel = %d/%d (=%.f%%: %.f%% too close + %.f%% not aligned)' % (nr_pulses_aligned, nr_pulses, perc_selected, perc_too_close, perc_outliers))
    return pulses_aligned, H, sel_locs, filtered_locs, pks_smoothed


def filter_pulses(pulses_aligned, H, sel_locs, filtered_locs, pks_smoothed, H_range, filter_std):
    nr_pulses = len(sel_locs)

    # To ensure the filtering is only applied on the selected range select these pulses only
    if H_range:
        if isinstance(H_range, (int, float)):
            idx_range = pks_smoothed >= H_range
        elif isinstance(H_range, (tuple, list, np.ndarray)):
            idx_range = (pks_smoothed >= H_range[0]) & (pks_smoothed < H_range[1])
        else:
            raise Exception('Please input H_range as integer or array-like')    
    else:
        idx_range = np.ones(len(pks_smoothed), dtype=bool)
    nr_pulses_range = idx_range.sum()

    # Compute mean and std of aligned pulses
    mean_aligned_pulse = np.mean(pulses_aligned[idx_range, :], axis=0)
    std_aligned_pulse = np.std(pulses_aligned[idx_range, :], axis=0)

    # Remove outliers
    max_aligned_pulse = mean_aligned_pulse + filter_std * std_aligned_pulse
    min_aligned_pulse = mean_aligned_pulse - filter_std * std_aligned_pulse
    outliers_above = np.all(np.less(pulses_aligned[idx_range, :], max_aligned_pulse), axis=1)
    outliers_below = np.all(np.greater(pulses_aligned[idx_range, :], min_aligned_pulse), axis=1)
    outliers = np.logical_and(outliers_above, outliers_below)
    pulses_aligned = np.vstack((pulses_aligned[~idx_range, :], pulses_aligned[idx_range][outliers, :]))
    filtered_locs = np.hstack((filtered_locs, sel_locs[idx_range][~outliers]))
    sel_locs = np.hstack((sel_locs[~idx_range], sel_locs[idx_range][outliers]))
    pks_smoothed = np.hstack((pks_smoothed[~idx_range], pks_smoothed[idx_range][outliers]))
    H = np.hstack((H[~idx_range], H[idx_range][outliers]))
    idx_range = np.hstack((idx_range[~idx_range], idx_range[idx_range][outliers]))

    # Print filtered number and percentage
    nr_final_pulses = len(sel_locs)
    nr_filtered = nr_pulses - nr_final_pulses
    perc_filtered = 100 * nr_filtered/nr_pulses
    nr_pulses_range_filtered = nr_pulses_range-nr_filtered
    perc_range = 100 * nr_pulses_range/nr_pulses
    perc_range_filtered = 100*nr_pulses_range_filtered/nr_pulses
    print('    N_range = %d/%d (=%.f%%: %.f%% out of range + %.1f%% filtered)' % (nr_pulses_range_filtered, nr_pulses, perc_range_filtered, 100 - perc_range, perc_filtered))
    return pulses_aligned, H, sel_locs, filtered_locs, pks_smoothed, idx_range


def noise_model(signal, pw, sff, nr_req_segments, sw, window):
    '''
    This function computes the average noise PSD from a given timestream
    '''
    # Initializing variables
    pw *= sff
    sw *= sff
    signal_length = len(signal)
    len_onesided = round(pw/ 2) + 1
    sxx_segments = np.zeros(len_onesided)

    # Smooth timestream for better pulse detection
    if sw:    
        smoothed_signal = fftconvolve(signal, window, mode='valid')
    else:
        smoothed_signal = signal

    # Compute std of the signal, twice, to set as a threshold for pulse detection in the noise segments
    std = np.std(smoothed_signal)
    threshold = np.round(5 * std, decimals=2)
    std = np.std(smoothed_signal[smoothed_signal < threshold])
    threshold = np.round(5 * std, decimals=2)
    
    # Compute the average noise PSD
    nr_good_segments = 0
    start = 0
    nr = 0
    stds = 0
    noise_segments = []
    while nr_good_segments < nr_req_segments:
        start = int(nr * pw)
        stop = int(start + pw)
        if stop >= signal_length:  # ensure that the next segment does not run out of the available data 
            print('     WARNING: only %d/%d noise segments obtained with max_bw=%d' % (nr_good_segments, nr_req_segments, pw))
            break
        next_segment = signal[start:stop]
        next_smoothed_segment = smoothed_signal[start:stop]
        nr_outliers = np.sum(next_smoothed_segment > threshold)
        if nr_outliers > 0:  # check whether there are pulses in the data
            nr += 1
        else:
            stds += np.std(next_smoothed_segment)
            freqs, sxx_segment = welch(next_segment, fs=int(sff*1e6), window='hamming', nperseg=pw, noverlap=None, nfft=None, return_onesided=True)
            sxx_segments += sxx_segment  # cumulatively add the PSDs of all the noise segments
            noise_segments.append(next_segment)
            nr_good_segments += 1
            nr += 1   
    if nr_good_segments == 0:
        raise Exception('No good noise segments found')
    sxx = sxx_segments / nr_good_segments  # compute the avarage PSD
    std = stds / nr_good_segments  # compute the avarage std
    threshold = np.round(5 * std, decimals=2)
    noise_segments = np.array(noise_segments).reshape((-1, pw))
    return freqs[1:len_onesided], sxx[1:len_onesided], threshold, noise_segments


def optimal_filter(pulses, pulse_model, sf, ssf_model, nxx):
    ''' 
    This function applies an optimal filter, i.e. a frequency weighted filter, to the pulses to extract a better estimate of the pulse heights
    '''
    # Initialize important variables 
    len_pulses = pulses.shape[-1]
    len_model = len(pulse_model)
    if len_pulses < len_model:
        len_onesided = round(len_pulses / 2) + 1
        ssf_pulses = 1
    else:
        len_onesided = round(len_pulses / ssf_model / 2) + 1
        ssf_pulses = ssf_model

    
    # Compute normalized pulse model
    norm_pulse_model = pulse_model / np.amax(pulse_model)

    # Step 1: compute psd and fft of normalized peak-model
    Mxx = psd(norm_pulse_model, sf*ssf_model)[1:len_onesided]
    Mf = fft(norm_pulse_model)[1:len_onesided] / ssf_model
    Mf_conj = Mf.conj()

    # Step 2: compute fft of all pulses
    Df = fft(pulses, axis=-1)[:, 1:len_onesided] / ssf_pulses
    Dxx = psd(pulses, sf*ssf_pulses)[:, 1:len_onesided]
    mean_Dxx = np.mean(Dxx, axis=0)

    # Step 3: obtain improved pulse height estimates
    numerator = Mf_conj * Df / nxx
    denominator = Mf_conj * Mf / nxx
    int_numerator = np.sum(numerator, axis=-1)
    int_denominator = np.sum(denominator, axis=-1)
    H = np.real(int_numerator / int_denominator)

    # Step 4: compute signal-to-noise resolving power
    NEP = (np.outer((1 / H)**2, (2 * nxx / Mxx)))**0.5
    dE = 2*np.sqrt(2*np.log(2)) * (np.sum(4 / NEP**2, axis=-1))**-0.5
    R_sn = np.mean(1 / dE)

    chi_sq = np.sum((Df - np.outer(H, Mf))**2 / nxx, axis=-1)

    return H, R_sn, mean_Dxx, chi_sq


def psd(array, fs):
    ''' 
    This function returns the PSD estimate using the fft for either a 1D or 2D array, see https://nl.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html
    '''
    # Obtain dimension of array
    ndim = array.ndim

    # For 1D arrays
    if ndim == 1:
        len_pulse = array.size
        len_onesided = round(len_pulse / 2) + 1
        fft_array = fft(array)
        fft_onesided = fft_array[:len_onesided]
        psd_array = 1 / (len_pulse * fs) * np.absolute(fft_onesided)**2
        psd_array[1:-1] *= 2
    # For 2D arrays    
    elif ndim == 2:
        len_pulse = array.shape[1]
        len_onesided = round(len_pulse / 2) + 1
        fft_array = fft(array, axis=-1)
        fft_onesided = fft_array[:, :len_onesided]
        psd_array = 1 / (len_pulse * fs) * np.absolute(fft_onesided)**2
        psd_array[:, 1:-1] *= 2
    else:  
        raise Exception("Sorry, only input n>0 pulses in a m x n array with m<=2")  
    return psd_array


def resolving_power(dist, histbin, range=None):
    ''' 
    This function obtains the resolving power of a distribution by means of a kernel density estimation
    '''
    
    # Limit the range of the KDE
    if range:
        if isinstance(range, (int, float)):
            dist = dist[dist>range]
        elif isinstance(range, (tuple, list, np.ndarray)):
            dist = dist[(dist > range[0]) & (dist < range[1])]
        else:
            raise Exception('Please input range as integer or array-like')
        
    # Check if distribution is not empty
    if dist.size == 0:
        raise Exception('Distribution is empty, check range')
    
    # Obtain pdf of distribution
    x = np.arange(np.amin(dist), np.amax(dist), histbin/10)
    nr_peaks = dist.size
    pdfkernel = gaussian_kde(dist, bw_method='scott')
    pdf = pdfkernel.evaluate(x)

    # Obtain index, value and x-position of the maximum of the distribution
    pdf_max = np.amax(pdf)
    pdf_max_idx = np.argmax(pdf)
    x_max = x[pdf_max_idx]

    # Find the left and right index and value of the pdf at half the maximum 
    hm = pdf_max / 2
   
    idx_right = (pdf > pdf_max / 4) & (x > x_max)
    pdf_right = pdf[idx_right]
    x_right = x[idx_right]
    if np.min(pdf_right) < hm < np.max(pdf_right):
        f_right = interp1d(pdf_right, x_right)(hm)
    else:
        f_right = np.max(pdf_right)
    
    idx_left = (pdf > pdf_max / 4) & (x < x_max) & (x > x_max - 2 * (f_right - x_max))
    x_left = x[idx_left]
    pdf_left = pdf[idx_left]
    if np.min(pdf_left) < hm < np.max(pdf_left):
        f_left = interp1d(pdf_left, x_left)(hm)
    else:
        f_left = np.min(pdf_left)

    # Compute the resolving power
    fwhm = f_right - f_left
    resolving_power = x_max / fwhm

    # Appropriately scale the pdf for plotting
    pdf = pdf * histbin * nr_peaks / (np.sum(pdf) * histbin/10)
        
    return resolving_power, pdf, x, x_max, fwhm


def fit_decaytime(pulse, pw, fit_T):
    ''' 
    This function returns the quasiparticle regeneration time, tau_qp by fitting a function y=a*exp(-x/tau_qp) to the tail of the pulse
    '''
    # Cut the tail from the pulse for fitting
    l = len(pulse)
    ssf = int(l / pw)
    t = np.linspace(0, pw, l)

    if isinstance(fit_T, (int, float)):
        fit_pulse = pulse[t>=fit_T]
        # fit_t = t[t>fit_T]
    elif isinstance(fit_T, (tuple, list, np.ndarray)):
        fit_pulse = pulse[(t>=fit_T[0]) & (t<fit_T[1])]
        # fit_t = t[(t>fit_T[0]) & (t<fit_T[1])]
    else:
        raise Exception('Please input fit_T as integer or array-like') 
    fit_x = np.arange(len(fit_pulse))

    # Obtain the optimal parameters from fit
    popt, pcov = curve_fit(exp_decay, fit_x, fit_pulse)

    # Obtain 1 std error on parameters
    perr = np.sqrt(np.diag(pcov))

    # Obtain tau_qp and error
    tau_qp = 1 / popt[1] / ssf
    dtau_qp = perr[1]

    return tau_qp, dtau_qp, popt


def exp_decay(x, a, b):
    ''' 
    This is a one-term exponential function used for aluminium KIDs: y=a*exp(-b * x)
    '''
    return a * np.exp(-b * x)



def get_window(type, tau):
    if type == 'box':
        M = int(tau / 2)
        y = windows.boxcar(M, sym=False)
        y /= np.sum(y)
    elif type == 'exp':
        M = int(tau*3)
        y = windows.exponential(M, center=0, tau=tau, sym=False)
        y /= np.sum(y) 
    else:
        raise Exception('Windowtype was given as %s. Please input a correct window type: "exp" or "box"' % type)
    return y[::-1]


def pulse_template(pw, rise_offset, tau):
    t = np.arange(pw)
    offset = np.zeros(rise_offset)
    pulse = np.exp(-t/tau)
    return np.hstack((offset, pulse))


def get_kid(dir_path, lt, wl, kid, date):
    data = '%sLT%s_%snm_KID%s*%s_data.txt' % (dir_path, str(lt), str(wl), str(kid), str(date))
    settings = '%sLT%s_%snm_KID%s*%s_settings.txt' % (dir_path, str(lt), str(wl), str(kid), str(date))
    data_path = glob.glob(data)
    settings_path = glob.glob(settings)
    if not data_path:
        raise Exception('Please correct kid path as no file was obtained: %s' % (data))
    if len(data_path) == 1:
        with open(data_path[0], 'rb') as file:
            kid = pickle.load(file)
        with open(settings_path[0], 'r') as file:
            settings = {}
            for line in file:
                (key, val) = re.split(":", line)
                kid[key] = val[:-1]                
    else:
        raise Exception('Multiple kids detected')
    return kid


def load_dictionary_from_file(file_path):
    try:
        with open(file_path, 'rb') as file:
            dictionary = pickle.load(file)
            return dictionary
    except FileNotFoundError:
        print(f"File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None