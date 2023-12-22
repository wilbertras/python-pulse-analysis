import functions as f
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, welch
import pickle
import matplotlib as mpl
import multiprocessing
mpl.style.use('bmh')
mpl.rcParams['axes.prop_cycle'] = mpl.rcParamsDefault['axes.prop_cycle']
mpl.rcParams.update({'font.size': 10})


# total = len(files)
#     chunk_size = total // multiprocessing.cpu_count()
#     chunks = [files[i:i + chunk_size] for i in range(0, total, chunk_size)]
#     with multiprocessing.Pool() as pool:
#         dfs = pool.map(parrallel_df, chunks)
#     dff = pd.concat(dfs, axis=0)

class MKID:
    def __init__(self, LT, wl, light_dir, dark_dir, kid_nr, pread, date, chuncksize=40):
        self.data = {}
        self.data['LT'] = LT
        self.data['wl'] = wl
        self.data['KID'] = kid_nr
        self.data['pread'] = pread
        self.data['date'] = date
        self.data['name'] = 'LT%d_%dnm_KID%d_P%d_%s' % (LT, wl, kid_nr, pread, date)
        self.chunckwise_peakmodel = False
        self.existing_peak_model = False
        self.light_files, self.light_info_file = f.get_bin_files(light_dir, kid_nr, pread)
        self.dark_files, _ = f.get_bin_files(dark_dir, kid_nr, pread)
        if len(self.dark_files) > chuncksize:
            self.dark_files = self.dark_files[:chuncksize]
        self.dark_amp, self.dark_phase = f.concat_vis(self.dark_files)
            
        self.chuncksize = chuncksize
        self.nr_segments = len(self.light_files)      
        if self.nr_segments <= self.chuncksize:
            self.amp, self.phase = f.concat_vis(self.light_files)
            self.chunckwise_peakmodel = False
        else:
            start = 0
            stop = self.chuncksize
            light_files_chunck = self.light_files[start:stop]
            self.amp, self.phase = f.concat_vis(light_files_chunck)
            self.chunckwise_peakmodel = True
        print('%d files obtained, chunckwise peakmodel is %s with chuncksize=%d' % (self.nr_segments, self.chunckwise_peakmodel, chuncksize))
        self.data['nr_segments'] = self.nr_segments
        
        if len(self.light_info_file) != 0:
            info = f.get_info(self.light_info_file[0])
        else:
            info = {'f0' : 0 , 'Q' : 0, 'Qc' : 0, 'Qi' : 0, 'S21_min' : 0, 'fs' : 0, 'T' : 0}
            print('No info file obtained')
        for x in info:
            self.data[x] = info[x]


    def initiate(self, settings, f, binsize=0.1, dpulse=10, plot_pulse=False, below=False):
        sf = settings['sf']
        response = settings['response']
        coord = settings['coord']
        pw = settings['pw']
        sw = settings['sw']
        window = settings['window']
        ssf = settings['ssf']
        buffer = settings['buffer']
        mph = settings['mph']
        mpp = settings['mpp']
        noise_mph = settings['noise_mph']
        noise_mpp = settings['noise_mpp']
        tlim = settings['tlim']
        filter_std = settings['filter_std']
        rise_offset = settings['rise_offset']
        binsize = settings['binsize']

        if ssf and ssf > 1:
            pass
        else:
            ssf = 1

        self.signal, self.dark_signal = f.coord_transformation(response, coord, self.phase, self.amp, self.dark_phase, self.dark_amp)
        
        pulses, H, sel_locs, filtered_locs, H_smoothed = f.peak_model(self.signal, mph, mpp, pw, sw, window, ssf, buffer, filter_std, rise_offset, plot_pulse, below)    

        t_idx = np.arange(int(tlim[0]*sf), int(tlim[1]*sf), 1)
        ylim = [-0.5, np.ceil(np.amax(self.signal[t_idx]))]
        xlim = [0, pw]
        t = np.linspace(tlim[0], tlim[1], len(t_idx))
        
        fig, axes = plt.subplot_mosaic("AABB;FECD", layout='constrained', figsize=(12, 6))
        fig.suptitle('Overview: %s' % (self.data['name']))
        axes["A"].plot(t, self.signal[t_idx], linewidth=0.5)  
        axes["A"].hlines(mph, *tlim, color='tab:red', lw=0.5)
        axes["A"].set_ylim(ylim)
        axes["A"].set_xlim(tlim)
        axes["A"].set_xlabel('time [s]')
        axes["A"].set_ylabel('response')
        axes["A"].set_title('Light')
        axes["B"].plot(t, self.dark_signal[t_idx], linewidth=0.5)
        axes["B"].hlines(noise_mph, *tlim, color='tab:red', lw=0.5)
        axes["B"].set_ylim(ylim)
        axes["B"].set_xlim(tlim)
        axes["B"].set_xlabel('time [s]')
        axes["B"].set_ylabel('response')
        axes["B"].set_title('Dark')
        
        if sw:  
            kernel = f.get_window(window, pw, sw)
            smoothed_signal = np.convolve(self.signal, kernel, mode='valid')
            smoothed_dark_signal = np.convolve(self.dark_signal, kernel, mode='valid')
            axes["A"].plot(t, smoothed_signal[t_idx], lw=0.5)
            axes["B"].plot(t, smoothed_dark_signal[t_idx], lw=0.5)
        else:
            smoothed_dark_signal = self.dark_signal
        
        plot_sel_idx = (sel_locs > tlim[0]*sf) & (sel_locs < tlim[1]*sf)
        plot_sel_locs = sel_locs[plot_sel_idx]
        plot_sel_H = H[plot_sel_idx]
        axes["A"].scatter(plot_sel_locs / sf, plot_sel_H, marker='v', c='None', edgecolors='tab:green', lw=0.5)
        if len(filtered_locs) != 0:
            plot_filtered_idx = (filtered_locs > tlim[0]*sf) & (filtered_locs < tlim[1]*sf)
            plot_filtered_locs = filtered_locs[plot_filtered_idx]
            axes["A"].scatter(plot_filtered_locs / sf, self.signal[plot_filtered_locs], marker='v', c='None', edgecolors='tab:red', lw=0.5)
        
        dark_locs, props = find_peaks(smoothed_dark_signal, height=noise_mph, prominence=noise_mpp)
        noise_heights = props['peak_heights']
        if len(dark_locs) != 0:
            plot_noise_idx = (dark_locs > tlim[0]*sf) & (dark_locs < tlim[1]*sf)
            plot_dark_locs = dark_locs[plot_noise_idx]
            plot_noise_heights = noise_heights[plot_noise_idx]
            axes["B"].scatter(plot_dark_locs / sf, plot_noise_heights, marker='v', c='None', edgecolors='tab:red', lw=0.5)
        
        t = np.linspace(0, pw-1, pw*ssf)
        axes["C"].plot(t, pulses[::dpulse, :].T, lw=0.25)
        axes["C"].set_ylim(ylim)
        axes["C"].set_xlim(xlim)
        axes["C"].set_xlabel('time [$\mu$s]')
        axes["C"].set_ylabel('response')
        axes["C"].set_title('Overlayed pulses')

        ylim = [1e-4, 3]
        axes["D"].semilogy(t, np.mean(pulses, axis=0))
        axes["D"].set_ylim(ylim)
        axes["D"].set_xlim(xlim)
        axes["D"].set_xlabel('time [$\mu$s]')
        axes["D"].set_ylabel('response')
        axes["D"].set_title('Average pulse on semilogy')
        binedges = np.arange(np.amin(H), np.amax(H), binsize)
        axes["E"].hist(H, bins=binedges)
        axes["E"].set_xlabel('response')
        axes["E"].set_ylabel('counts')
        axes["E"].set_title('Pulse heights')
        binedges = np.arange(mph, np.amax(H_smoothed), binsize)
        axes["F"].hist(H_smoothed, bins=binedges, facecolor='tab:orange')
        axes["F"].axvline(mph, c='tab:red')
        axes["F"].set_xlabel('response')
        axes["F"].set_ylabel('counts')
        axes["F"].set_title('Smoothed pulse heights')


    def overview(self, settings, f, redo_peak_model=False, save=False, figpath=''):
        sf = settings['sf']
        response = settings['response']
        coord = settings['coord']
        pw = settings['pw']
        sw = settings['sw']
        window = settings['window']
        ssf = settings['ssf']
        buffer = settings['buffer']
        mph = settings['mph']
        mpp = settings['mpp']
        noise_mph = settings['noise_mph']
        noise_mpp = settings['noise_mpp']
        nr_noise_segments = settings['nr_noise_segments']
        binsize = settings['binsize']
        range = settings['range']
        fit_T = settings['fit_T']
        max_bw = settings['max_bw']
        tlim = settings['tlim']
        filter_std = settings['filter_std']
        rise_offset = settings['rise_offset']

        self.signal, self.dark_signal = f.coord_transformation(response, coord, self.phase, self.amp, self.dark_phase, self.dark_amp)
        first_sec = int(1 * sf)
        self.data['signal'] = self.signal[0:first_sec]
        self.data['dark_signal'] = self.dark_signal[0:first_sec]

        fxx, nxx, _, _ = f.noise_model(self.dark_signal, pw, sf, ssf, nr_noise_segments, noise_mph, noise_mpp, sw)
        
        Nfxx, Nxx, dark_locs, dark_photon_rate = f.noise_model(self.dark_signal, max_bw, sf, None, nr_noise_segments, noise_mph, noise_mpp, sw)
        self.data['dark_locs'] = dark_locs

        if self.existing_peak_model==False or (self.existing_peak_model==True and redo_peak_model==True): 
            if self.chunckwise_peakmodel:
                start = 0
                stop = self.chuncksize
                chunck = self.chuncksize
                
                pulses = []
                H = []
                sel_locs = []
                filtered_locs = []
                H_smoothed = []
                while stop <= self.nr_segments:
                    light_files_chunck = self.light_files[start:stop]
                    if start == 0:
                        amp, phase = self.amp, self.phase
                    else:
                        amp, phase = f.concat_vis(light_files_chunck)

                    signal = f.coord_transformation(response, coord, phase, amp)

                    pulses_chunck, H_chunck, sel_locs_chunck, filtered_locs_chunck, H_smoothed_chunck = f.peak_model(signal, mph, mpp, pw, sw, window, ssf, buffer, filter_std, rise_offset)
                    
                    if len(sel_locs_chunck) != 0:
                        sel_locs_chunck += int(start * sf)
                        pulses.append(pulses_chunck)
                        H.append(H_chunck)
                        sel_locs.append(sel_locs_chunck)
                        filtered_locs.append(filtered_locs_chunck)
                        H_smoothed.append(H_smoothed_chunck)
                    if len(filtered_locs_chunck) != 0:
                        filtered_locs_chunck += int(start * sf)              

                    start = stop
                    stop += chunck
                    if start >= self.nr_segments:
                        stop = self.nr_segments + 1
                    elif stop > self.nr_segments:
                        stop = self.nr_segments

                pulses = np.concatenate(pulses)
                H = np.concatenate(H)
                sel_locs = np.concatenate(sel_locs)
                filtered_locs = np.concatenate(filtered_locs)
                H_smoothed = np.concatenate(H_smoothed)
            else:  
                pulses, H, sel_locs, filtered_locs, H_smoothed = f.peak_model(self.signal, mph, mpp, pw, sw, window, ssf, buffer, filter_std, rise_offset)  
            self.data['pulses'] = pulses
            self.data['sel_locs'] = sel_locs
            self.data['filtered_locs'] = filtered_locs
            self.data['H'] = H
            self.data['H_smoothed'] = H_smoothed
            self.existing_peak_model = True
        else:
            pulses = self.data['pulses']
            H = self.data['H']
            sel_locs = self.data['sel_locs']
            filtered_locs = self.data['filtered_locs']
            H_smoothed = self.data['H_smoothed']
        
        if range:
            if isinstance(range, (int, float)):
                idx_range = H_smoothed > range
            elif isinstance(range, (tuple, list)):
                idx_range = (H_smoothed > range[0]) & (H_smoothed < range[1])
            else:
                raise Exception('Please input range as integer or array-like')    
        else:
            idx_range = np.ones(len(H_smoothed), dtype=bool)
        
        _, dark_H, _, _, dark_H_smoothed = f.peak_model(self.dark_signal, noise_mph, noise_mpp, pw, sw, window, ssf, buffer, filter_std, rise_offset)  

        ## Mean pulse
        pulses_range = pulses[idx_range, :]
        H_range = H[idx_range]
        mean_pulse = np.mean(pulses_range, axis=0)
        sxx = f.psd(mean_pulse, sf*ssf)
        
        ## Determine some pulse statistics
        nr_sel_pulses = len(sel_locs)
        if nr_sel_pulses == 0:
            raise Exception("No pulses selected")
        nr_rej_pulses = len(filtered_locs)
        nr_det_pulses = nr_sel_pulses + nr_rej_pulses
        rej_perc = 100 * (1 - nr_sel_pulses / nr_det_pulses)
        photon_rate = nr_det_pulses / self.nr_segments

        ## Optimal filtering and resolving powers
        H_opt, R_sn, mean_dxx = f.optimal_filter(pulses_range, mean_pulse, sf, ssf, nxx)
        mean_H_opt = np.mean(H_opt)
        R, _, _ = f.resolving_power(H_range, binsize)
        R_opt, pdf_y, pdf_x = f.resolving_power(H_opt, binsize)
        R_i = 1 / np.sqrt(1 / R_opt**2 - 1 / R_sn**2)
        
        ## Fit lifetime
        tau_qp, dtau_qp, popt = f.fit_decaytime(mean_pulse, pw, fit_T)
        if isinstance(fit_T, (int, float)):
            plot_x = np.linspace(fit_T, pw, (pw - fit_T) * ssf)
        elif isinstance(fit_T, (tuple, list)):
            plot_x = np.linspace(fit_T[0], fit_T[1], (fit_T[1] - fit_T[0]) * ssf)
        fit_x = np.arange(len(plot_x))
        fit_y = f.exp_decay(fit_x, *popt)

        ## Add data and settings to MKID object
        self.data['mean_pulse'] = mean_pulse
        self.data['R'] = R
        self.data['Ropt'] = R_opt
        self.data['Ri'] = R_i
        self.data['Rsn'] = R_sn
        self.data['dark_H'] = dark_H
        self.data['dark_H_smoothed'] = dark_H_smoothed
        self.data['idx_H_range'] = idx_range
        self.data['Hopt'] = H_opt
        self.data['<Hopt>'] = mean_H_opt
        self.data['Nph'] = photon_rate
        self.data['Nph_dark'] = dark_photon_rate
        self.data['rej.'] = rej_perc
        self.data['pdfx'] = pdf_x
        self.data['pdfy'] = pdf_y
        self.data['fxx'] = fxx
        self.data['sxx'] = sxx
        self.data['nxx'] = nxx
        self.data['mean_dxx'] = mean_dxx
        self.data['Nfxx'] = Nfxx
        self.data['Nxx'] = Nxx
        self.data['tqp'] = tau_qp
        self.data['fitx'] = plot_x
        self.data['fity'] = fit_y
        self.data['sel_locs'] = sel_locs
        self.data['filtered_locs'] = filtered_locs
        self.data['dark_locs'] = dark_locs
        self.settings = settings

        ## Plot overview
        self.plot_overview()

        ## Save data
        if save:
            self.save(figpath)


    def plot_overview(self):
        sw = self.settings['sw']
        tlim = self.settings['tlim']
        binsize = self.settings['binsize']
        fig, axes = plt.subplot_mosaic("AABB;CDEF;GHIF;JJJJ", layout='constrained', figsize=(12, 8))
        fig.suptitle('Overview: %s' % (self.data['name']))
        self.plot_timestream(axes['A'], 'light', tlim)
        self.plot_timestream(axes['B'], 'dark', tlim)
        if sw:
            self.plot_hist(axes['C'], 'smoothed', binsize / 2)
        self.plot_hist(axes['D'], 'unsmoothed', binsize)
        self.plot_hist(axes['E'], 'optimal filter', binsize)
        self.plot_psds(axes['F'])
        self.plot_stacked_pulses(axes['G'])
        self.plot_mean_pulse(axes['H'], type='lin')
        self.plot_mean_pulse(axes['I'], type='log')
        self.plot_table(axes['J'])


    def plot_timestream(self, ax, type, tlim):
        sf = self.settings['sf']
        pw = self.settings['pw']
        mph = self.settings['mph']
        sw = self.settings['sw']
        H = self.data['H']
        window = self.settings['window']

        if type == 'light':
            signal = self.signal
            locs = self.data['sel_locs']
            filtered_locs = self.data['filtered_locs']
            ax.set_title('Light timestream')
        elif type == 'dark':   
            signal = self.dark_signal 
            locs = self.data['dark_locs']
            ax.set_title('Dark timestream')

        t_idx = np.arange(int(tlim[0]*sf), int(tlim[1]*sf), 1)
        ylim = [-0.5, np.ceil(np.amax(signal[t_idx]))]
        t = np.linspace(tlim[0], tlim[1], len(t_idx))

        ax.plot(t, signal[t_idx], linewidth=0.5, label='timestream')  
        if sw:  
            kernel = f.get_window(window, pw, sw)
            smoothed_signal = np.convolve(signal, kernel, mode='valid')
            ax.plot(t, smoothed_signal[t_idx], lw=0.5, label='smoothed')
        ax.axhline(mph, color='tab:red', lw=0.5, label='mph')
        
        if type == 'light':
            plot_sel_idx = (locs > tlim[0]*sf) & (locs < tlim[1]*sf)
            plot_sel_locs = locs[plot_sel_idx]
            plot_sel_H = H[plot_sel_idx]
            ax.scatter(plot_sel_locs / sf, plot_sel_H, marker='v', c='None', edgecolors='tab:green', lw=0.5, label='sel. pulses')
            plot_filtered_idx = (filtered_locs > tlim[0]*sf) & (filtered_locs < tlim[1]*sf)
            plot_filtered_locs = filtered_locs[plot_filtered_idx]
            ax.scatter(plot_filtered_locs / sf, self.signal[plot_filtered_locs], marker='v', c='None', edgecolors='tab:red', lw=0.5, label='del. pulses')
        elif type == 'dark':
            plot_dark_idx = (locs > tlim[0]*sf) & (locs < tlim[1]*sf)
            plot_dark_locs = locs[plot_dark_idx]
            ax.scatter(plot_dark_locs / sf, self.signal[plot_dark_locs], marker='v', c='None', edgecolors='tab:red', lw=0.5)
        ax.set_ylim(ylim)
        ax.set_xlim(tlim)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('response')
        ax.legend(ncols=3)


    def plot_stacked_pulses(self, ax):
        pw = self.settings['pw']
        pulses = self.data['pulses']
        nr_pulses = pulses.shape[0]
        len_pulses = pulses.shape[-1]
        t = np.linspace(0, pw-1, len_pulses)
        nr2plot = 1000
        if nr2plot > nr_pulses:
            every = 1
        else:
            every = int(np.round(nr_pulses / nr2plot))           
        pulses2plot = pulses[::every, :].T
        ax.plot(t, pulses2plot, lw=0.25)
        ax.set_xlim([0, pw-1])
        ax.set_xlabel('time [$\mu$s]')
        ax.set_ylabel('response')
        ax.set_title('%d overlayed pulses' % pulses2plot.shape[-1])
    

    def plot_psds(self, ax):
        pw = self.settings['pw']
        sxx = self.data['sxx']
        fxx = self.data['fxx']
        nxx = self.data['nxx']
        mean_dxx = self.data['mean_dxx']
        onesided = round(pw / 2) + 1
        ax.semilogx(fxx[:onesided], 10*np.log10(sxx[:onesided]), label='psd(avg. pulse)')
        ax.semilogx(fxx[:onesided], 10*np.log10(nxx[:onesided]), label='avg. psd(noise)')
        ax.semilogx(fxx[:onesided], 10*np.log10(mean_dxx[:onesided]), label='avg. psd(pulses)')
        ax.set_ylim([-100, 10*np.log10(np.amax(sxx))])
        ax.grid(which='major', lw=0.5)
        ax.grid(which='minor', lw=0.2)
        ax.xaxis.get_major_locator().set_params(numticks=99)
        ax.xaxis.get_minor_locator().set_params(numticks=99, subs=np.arange(2, 10, 2)*.1)
        ax.legend()
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('PSD [dBc/Hz]')
        ax.set_title('Optimal filter models')
    

    def plot_mean_pulse(self, ax, type):
        pw = self.settings['pw']
        ssf = self.settings['ssf']
        mean_pulse = self.data['mean_pulse']
        fitx = self.data['fitx']
        fity = self.data['fity']
        t = np.linspace(0, pw-1, pw*ssf)
        if type == 'lin': 
            ax.plot(t, mean_pulse)
            ax.plot(fitx, fity, ls='--', label='fit')
            ax.set_title('Average pulse')
        elif type == 'log':
            ax.semilogy(t, mean_pulse)
            ax.semilogy(fitx, fity, ls='--', label='fit')
            ax.set_title('Average pulse semilog')
        ax.set_xlim([0, pw])
        ax.set_xlabel('time [$\mu$s]')
        ax.set_ylabel('response')
        ax.legend()
        

    def plot_table(self, ax):
        table_results = ['T', 'Q', 'Qi', 'Qc', 'Nph_dark', 'Nph', 'rej.', '<Hopt>', 'R', 'Ropt', 'Ri', 'Rsn', 'tqp']
        table_formats = ['%.1f', '%.1e', '%.1e', '%.1e', '%.1f', '%.f', '%.f', '%.1f', '%.1f', '%.1f', '%.1f', '%.1f', '%.f']
        results_text = [[table_formats[i] % self.data[table_results[i]] for i in np.arange(len(table_results))]]
        
        the_table = ax.table(cellText=results_text,
                    rowLabels=['results'],
                    colLabels=table_results,
                    loc='center')
        _ = ax.axis('off')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)


    def plot_hist(self, ax, type, binsize):
        mph = self.settings['mph']
        noise_mph = self.settings['noise_mph']
        H_smoothed = self.data['H_smoothed']
        H = self.data['H']
        Hopt = self.data['Hopt']
        range = self.settings['range']
        bin_min = np.amin((np.amin(H), np.amin(Hopt)))
        bin_max = np.amax((np.amax(H), np.amax(Hopt)))
        bin_edges = np.arange(bin_min, bin_max, binsize)
        idx_range = self.data['idx_H_range']
        if type=='smoothed':
            ax.hist(H_smoothed[idx_range], bins=bin_edges, label='sel.', color='tab:orange', alpha=0.5)
            ax.hist(H_smoothed[~idx_range], bins=bin_edges, label='del.', color='tab:grey', alpha=0.5) 
            if range:
                if isinstance(range, (int, float)):
                    ax.axvline(range, c='r', lw=0.5, ls='--', label='sel. range')
                elif isinstance(range, (tuple, list)):
                    for line in range:
                        ax.axvline(line, c='r', lw=0.5, ls='--', label='sel. range')
            ax.axvline(mph, c='r', lw=0.5, label='mph')
            ax.set_title('Smoothed heights')    
        elif type=='unsmoothed':
            ax.hist(H[idx_range], bins=bin_edges, label='sel.', color='tab:blue', alpha=0.5)
            ax.hist(H[~idx_range], bins=bin_edges, label='del.', color='tab:grey', alpha=0.5)
            ax.axvline(mph, c='r', lw=0.5, label='mph')
            ax.set_title('Unsmoothed heights')
        elif type == 'optimal filter':
            pdfx = self.data['pdfx']
            pdfy = self.data['pdfy']
            ax.hist(H[idx_range], bins=bin_edges, label='original', color='tab:blue', alpha=0.5)
            ax.hist(Hopt, bins=bin_edges, color='tab:green', alpha=0.5, label='opt. filter')
            ax.plot(pdfx, pdfy, c='tab:green', label='KDE')
            ax.set_title('Optimal filter heights')
        elif type == 'dark':
            dark_H = self.data['dark_H']
            ax.hist(dark_H, label='dark', color='tab:green', alpha=0.5)
            ax.hist(H[idx_range], bins=bin_edges, label='light', color='tab:blue', alpha=0.5)
            ax.axvline(noise_mph, c='r', lw=0.5)
            ax.set_title('Dark heights')
        ax.set_xlim([0, bin_max])   
        ax.legend()
        ax.set_xlabel('pulse height')
        ax.set_ylabel('counts')

    def plot_psd_noise(self, ax):
        Nxx = self.data['Nxx']
        Nfxx = self.data['Nfxx']
        ax.semilogx(Nfxx[1:], 10*np.log10(Nxx[1:]))
        ax.set_ylim([-100, 10*np.log10(np.amax(Nxx))])
        ax.grid(which='major', lw=0.5)
        ax.grid(which='minor', lw=0.2)
        ax.xaxis.get_major_locator().set_params(numticks=99)
        ax.xaxis.get_minor_locator().set_params(numticks=99, subs=np.arange(2, 10, 2)*.1)
        ax.legend()
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('PSD [dBc/Hz]')
        ax.set_title('PSD dark')
        
    def save(self, figpath):
        fname = figpath + self.data['name']
        plt.savefig(fname + '_overview.png')
        plt.savefig(fname + '_overview.svg')
        with open(fname + '_settings.txt','w') as f: 
            for key, value in self.settings.items():
                f.write('%s:%s\n' % (key, value))
        with open(fname + '_data.txt', 'wb') as file:
            pickle.dump(self.data, file)
    
