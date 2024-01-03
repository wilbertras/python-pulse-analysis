import multiprocessing
import functions as f
import numpy as np
import itertools

def square(x):
    return x * x

def multithread_peakmodel(settings, chunck, loc_offset):
    response = settings['response']
    coord = settings['coord']
    pw = settings['pw']
    sw = settings['sw']
    window = settings['window']
    ssf = settings['ssf']
    buffer = settings['buffer']
    mph = settings['mph']
    mpp = settings['mpp']
    filter_std = settings['filter_std']
    rise_offset = settings['rise_offset']

    amp, phase = f.concat_vis(chunck)
    signal = f.coord_transformation(response, coord, phase, amp)
    result = f.peak_model(signal, mph, mpp, pw, sw, window, ssf, buffer, filter_std, rise_offset)
    sel_locs = result[2]
    filtered_locs = result[3]
    if len(sel_locs) != 0: 
        result[2][0] += loc_offset
    if len(filtered_locs) != 0:
        result[3][0] = loc_offset  
    return result

if __name__ == "__main__":
    light_dir = "D:/Data/LT218Chip1_BF_20221025_MIR8_5/12KIDs LN2 load - long/TD_Power" 
    dark_dir = "D:/Data/LT218Chip1_BF_20221025_MIR8_5/12KIDs LN2 load - long/TD_Power" 
    LT = 218
    wavelength = 8500
    kid_nr = 21
    pread = 103
    comment = '20221025_LN2load'
    chuncksize = 40
    light_files, light_info_file = f.get_bin_files(light_dir, kid_nr, pread)
    psc = 0.045
    nsc = 0.12
    settings = {'sf':1e6,'coord':'smith','response':'X','pw':256,'sw':30,'window':'box','ssf':1,'buffer':0.25,'mph':psc,'mpp':psc,'noise_mph':nsc,'noise_mpp':nsc,
                'nr_noise_segments':100,'binsize':0.02,'H_range':0.12,'fit_T':[50, 150],'max_bw':10000,'tlim':[0, 1],'filter_std':5,'rise_offset':0.1} 
    nr_segments = len(light_files)
    nr_threads = int(multiprocessing.cpu_count())
    chunk_size = int(nr_segments // nr_threads)
    total_processed = int(chunk_size * nr_threads)
    chunks = [light_files[i:i + chunk_size] for i in range(0, total_processed, chunk_size)]
    loc_offsets = np.arange(0, total_processed, chunk_size, dtype=int)

    with multiprocessing.Pool() as pool:
        results = pool.starmap(multithread_peakmodel, zip(itertools.repeat(settings), chunks, loc_offsets))

    pulses = np.concatenate([sublist[0] for sublist in results])
    H = np.concatenate([sublist[1] for sublist in results])
    sel_locs = np.concatenate([sublist[2] for sublist in results])
    filtered_locs = np.concatenate([sublist[3] for sublist in results])
    H_smoothed = np.concatenate([sublist[4] for sublist in results])
    print(len(pulses))
