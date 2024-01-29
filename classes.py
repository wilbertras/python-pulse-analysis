import functions as f
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, welch
import pickle
import matplotlib as mpl
import multiprocessing
import itertools
mpl.style.use('bmh')
mpl.rcParams['axes.prop_cycle'] = mpl.rcParamsDefault['axes.prop_cycle']
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times New Roman'
mpl.rcParams.update({'font.size': 10})
mpl.rcParams["mathtext.default"] = 'rm'


class MKID:
    def __init__(self, *args, file_path=None):
        if not args:
            self.load_mkid(file_path)
        elif file_path is None:
            [LT, wl, light_dir, dark_dir, kid_nr, pread, date, chuncksize] = args
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
            self.nr_dark_segments = len(self.dark_files)      
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

            if len(self.light_info_file) != 0:
                info = f.get_info(self.light_info_file[0])
            else:
                info = {'f0' : 0 , 'Q' : 0, 'Qc' : 0, 'Qi' : 0, 'S21_min' : 0, 'fs' : 0, 'T' : 0}
                print('No info file obtained')
            for x in info:
                self.data[x] = info[x]


    def overview(self, settings, f, max_chuncks=None, redo_peak_model=False, plot_pulses=False, save=False, figpath=''):
        self.settings = settings
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
        nr_noise_segments = settings['nr_noise_segments']
        binsize = settings['binsize']
        H_range = settings['H_range']
        fit_T = settings['fit_T']
        max_bw = settings['max_bw']
        tlim = settings['tlim']
        filter_std = settings['filter_std']
        rise_offset = settings['rise_offset']

        self.signal, self.dark_signal = f.coord_transformation(response, coord, self.phase, self.amp, self.dark_phase, self.dark_amp)
        first_sec = int(1 * sf)
        self.data['signal'] = self.signal[0:first_sec]
        self.data['dark_signal'] = self.dark_signal[0:first_sec]

        fxx, nxx, dark_threshold = f.noise_model(self.dark_signal, pw, sf, ssf, nr_noise_segments, sw)
        self.data['dark_threshold'] = dark_threshold
        if mph == None:
            mph = dark_threshold
            mpp = mph
            self.settings['mph'] = mph
            self.settings['mpp'] = mpp

        Nfxx, Nxx, _ = f.noise_model(self.dark_signal, max_bw, sf, None, nr_noise_segments, sw)

        if self.existing_peak_model==False or (self.existing_peak_model==True and redo_peak_model==True): 
            if max_chuncks:
                if max_chuncks > 1:
                    self.chunckwise_peakmodel = True
                else:
                    self.chunckwise_peakmodel = False
            elif max_chuncks is None:
                self.chunckwise_peakmodel = True
            if self.chunckwise_peakmodel:
                start = 0
                stop = self.chuncksize
                chunck = self.chuncksize
                pulses = []
                H = []
                sel_locs = []
                filtered_locs = []
                H_smoothed = []

                if max_chuncks:
                    self.nr_segments = max_chuncks * chunck

                while stop <= self.nr_segments:
                    light_files_chunck = self.light_files[start:stop]
                    if start == 0:
                        amp, phase = self.amp, self.phase
                    else:
                        amp, phase = f.concat_vis(light_files_chunck)

                    signal = f.coord_transformation(response, coord, phase, amp)

                    if plot_pulses and start==0:
                        pass
                    else:
                        plot_pulses = False

                    pulses_chunck, H_chunck, sel_locs_chunck, filtered_locs_chunck, H_smoothed_chunck = f.peak_model(signal, mph, mpp, pw, sw, window, ssf, buffer, filter_std, rise_offset, plot_pulse=plot_pulses)
                    
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
                pulses, H, sel_locs, filtered_locs, H_smoothed = f.peak_model(self.signal, mph, mpp, pw, sw, window, ssf, buffer, filter_std, rise_offset, plot_pulse=plot_pulses)  
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
        
        if H_range:
            if isinstance(H_range, (int, float)):
                idx_range = H_smoothed > H_range
            elif isinstance(H_range, (tuple, list)):
                idx_range = (H_smoothed > H_range[0]) & (H_smoothed < H_range[1])
            else:
                raise Exception('Please input H_range as integer or array-like')    
        else:
            idx_range = np.ones(len(H_smoothed), dtype=bool)
        
        _, dark_H, sel_dark_locs, filtered_dark_locs, dark_H_smoothed = f.peak_model(self.dark_signal, mph, mpp, pw, sw, window, ssf, buffer, filter_std, rise_offset)  
        dark_locs = np.hstack((sel_dark_locs, filtered_dark_locs))
        self.data['dark_locs'] = dark_locs

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
        photon_rate_range = len(H_range) / self.nr_segments
        dark_photon_rate = len(dark_locs) / self.nr_dark_segments

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
        self.data['Nph_range'] = photon_rate_range
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
        self.data['nr_segments'] = self.nr_segments
        self.data['nr_dark_segments'] = self.nr_dark_segments
        self.data['settings'] = settings

        ## Plot overview
        self.plot_overview()

        ## Save data
        if save:
            self.save(figpath)


    def plot_overview(self):
        sw = self.settings['sw']
        tlim = self.settings['tlim']
        binsize = self.settings['binsize']
        fig, axes = plt.subplot_mosaic("AABB;CDEF;GHIJ;KKKK", layout='constrained', figsize=(18, 8))
        fig.suptitle('Overview: %s' % (self.data['name']))
        self.plot_timestream(axes['A'], 'light', tlim)
        self.plot_timestream(axes['B'], 'dark', tlim)
        if sw:
            self.plot_hist(axes['C'], 'dark smoothed', binsize)
            self.plot_hist(axes['D'], 'smoothed', binsize * np.amax(self.data['H_smoothed'])/np.amax(self.data['H']))
        self.plot_hist(axes['E'], 'unsmoothed', binsize)
        self.plot_hist(axes['F'], 'optimal filter', binsize)
        
        self.plot_psds(axes['J'])
        self.plot_stacked_pulses(axes['G'])
        self.plot_mean_pulse(axes['H'], type='lin')
        self.plot_mean_pulse(axes['I'], type='log')
        self.plot_table(axes['K'])


    def plot_timestream(self, ax, type, tlim=(0, 1)):
        sf = self.settings['sf']
        pw = self.settings['pw']
        mph = self.settings['mph']
        sw = self.settings['sw']
        H = self.data['H']
        window = self.settings['window']

        if type == 'light':
            signal = self.data['signal']
            locs = self.data['sel_locs']
            filtered_locs = self.data['filtered_locs']
            ax.set_title('Light timestream')
        elif type == 'dark':   
            signal = self.data['dark_signal'] 
            locs = self.data['dark_locs']
            ax.set_title('Dark timestream')
        if int(tlim[-1]*sf) > len(signal):
            tlim = (0, 1)
        plot_locs_idx = (locs > tlim[0]*sf) & (locs < tlim[1]*sf)
        plot_locs = locs[plot_locs_idx]

        t_idx = np.arange(int(tlim[0]*sf), int(tlim[1]*sf), 1)
        ylim = [-0.5, np.ceil(np.amax(signal[t_idx]))]
        t = t_idx / sf

        ax.plot(t, signal[t_idx], linewidth=0.5, label='timestream')  
        if sw:  
            kernel = f.get_window(window, pw, sw)
            smoothed_signal = np.convolve(signal, kernel, mode='same')
            ax.plot(t, smoothed_signal[t_idx], lw=0.5, label='smoothed')
        
        if type == 'light':
            if len(plot_locs_idx):
                plot_sel_H = H[plot_locs_idx]
                ax.scatter(plot_locs / sf, plot_sel_H, marker='v', c='None', edgecolors='tab:green', lw=0.5, label='sel. pulses')
            plot_filtered_idx = (filtered_locs > tlim[0]*sf) & (filtered_locs < tlim[1]*sf)
            if len(plot_filtered_idx):
                plot_filtered_locs = filtered_locs[plot_filtered_idx]
                plot_filtered_H = signal[plot_filtered_locs]
                ax.scatter(plot_filtered_locs / sf, plot_filtered_H, marker='v', c='None', edgecolors='tab:red', lw=0.5, label='del. pulses')
        elif type == 'dark':
            if len(plot_locs):
                plot_dark_H = signal[plot_locs]
                ax.scatter(plot_locs / sf, plot_dark_H, marker='v', c='None', edgecolors='tab:red', lw=0.5)
        ax.axhline(mph, color='tab:red', lw=0.5, label='$\it{mph}=%.2f$' % (mph))
        ax.set_ylim(ylim)
        ax.set_xlim(tlim)
        ax.set_xlabel('$\it{t}$ $[s]$')
        ax.set_ylabel('$response$')
        ax.legend(loc='lower center', ncols=5)


    def plot_stacked_pulses(self, ax):
        pw = self.settings['pw']
        pulses = self.data['pulses']
        nr_pulses = pulses.shape[0]
        len_pulses = pulses.shape[-1]
        t = np.linspace(0, pw-1, len_pulses)
        nr2plot = 100
        if nr2plot > nr_pulses:
            every = 1
        else:
            every = int(np.round(nr_pulses / nr2plot))           
        pulses2plot = pulses[::every, :].T
        ax.plot(t, pulses2plot, lw=0.25)
        ax.set_xlim([0, pw-1])
        ax.set_xlabel('$\it{t}$ $[\mu s]$')
        ax.set_ylabel('$response$')
        ax.set_title('%d overlayed pulses' % pulses2plot.shape[-1])
    

    def plot_psds(self, ax):
        pw = self.settings['pw']
        sxx = self.data['sxx']
        fxx = self.data['fxx']
        nxx = self.data['nxx']
        mean_dxx = self.data['mean_dxx']
        onesided = round(pw / 2) + 1
        ax.semilogx(fxx[1:onesided], 10*np.log10(sxx[1:onesided]), label='$\it M(f)$')
        ax.semilogx(fxx[1:onesided], 10*np.log10(nxx[1:onesided]), label='$\it N(f)$')
        ax.semilogx(fxx[1:onesided], 10*np.log10(mean_dxx[1:onesided]), label='$\it D(f)$')
        ax.set_ylim([-100, 10*np.log10(np.amax(sxx))])
        ax.set_xlim([fxx[1], .5*self.settings['sf']])
        ax.grid(which='major', lw=0.5)
        ax.grid(which='minor', lw=0.2)
        ax.xaxis.get_major_locator().set_params(numticks=99)
        ax.xaxis.get_minor_locator().set_params(numticks=99, subs=np.arange(2, 10, 2)*.1)
        ax.legend()
        ax.set_xlabel('$\it{f}$ $[Hz]$')
        ax.set_ylabel('$PSD$ $[dBc/Hz]$')
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
        ax.set_xlabel('$\it{t}$ $[\mu s]$')
        ax.set_ylabel('$response$')
        ax.legend()
        

    def plot_table(self, ax):
        table_results = ['T', 'nr_segments', 'Q', 'Qi', 'Qc', 'Nph_dark', 'Nph_range', 'rej.', '<Hopt>', 'R', 'Ropt', 'Ri', 'Rsn', 'tqp']
        col_labels = ['$\it{T}$ $[K]$', '$\it{t}$ $[s]$', '$\it{Q}$', '$\it{Q_i}$', '$\it{Q_c}$', '$\it{N}_{ph}^{dark}$ $[cps]$', '$\it{N}_{ph}^{sel}$ $[cps]$', 'rej. [%]', '$<\it{H}_{opt}>$ $[rad]$', '$\it{R}$', '$\it{R}_{opt}$', '$\it{R}_i$', '$\it{R}_{sn}$', '$\it{\\tau}_{qp}$ $[\mu s]$']
        table_formats = ['%.1f', '%d', '%.1e', '%.1e', '%.1e', '%.1f', '%.f', '%.f', '%.1f', '%.1f', '%.1f', '%.1f', '%.1f', '%.f']
        results_text = [[table_formats[i] % self.data[table_results[i]] for i in np.arange(len(table_results))]] 
        the_table = ax.table(cellText=results_text,
                    rowLabels=['results'],
                    colLabels=col_labels,
                    loc='center')
        _ = ax.axis('off')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(9)


    def plot_hist(self, ax, type, binsize):
        mph = self.settings['mph']
        dark_threshold = self.data['dark_threshold']
        idx_range = self.data['idx_H_range']
        H_range = self.settings['H_range']
        if H_range:
            if isinstance(H_range, (int, float)):
                lim = H_range 
            elif isinstance(H_range, (tuple, list)):
                if H_range[0] == mph:
                    lim = H_range[1]
                else:
                    lim = H_range[0]
        else:
            lim = mph
        nr_bins = int(np.round((lim - mph) / binsize))
        if nr_bins == 0:
            new_binsize = binsize
        else:
            new_binsize = (lim-mph) / nr_bins
            self.settings['binsize'] = new_binsize
        if type=='smoothed':
            H_smoothed = self.data['H_smoothed']
            bin_edges = np.arange(mph, np.amax(H_smoothed)+new_binsize, new_binsize)
            ax.hist(H_smoothed[idx_range], bins=bin_edges, label='sel.', color='tab:orange', alpha=0.5)
            ax.hist(H_smoothed[~idx_range], bins=bin_edges, label='del.', color='tab:grey', alpha=0.5) 
            if H_range:
                if isinstance(H_range, (int, float)):
                    ax.axvline(H_range, c='r', lw=0.5, ls='--', label='$limit$')
                elif isinstance(H_range, (tuple, list)):
                    ax.axvline(H_range[0], c='r', lw=0.5, ls='--', label='$limit$')
                    ax.axvline(H_range[1], c='r', lw=0.5, ls='--')
            ax.axvline(mph, c='r', lw=0.5, label='$\it{mph}=%.2f$' % (mph))
            bin_max = np.amax(H_smoothed)
            ax.set_xlim([0, bin_max])
            ax.set_title('Smoothed heights')    
        elif type=='unsmoothed':
            H = self.data['H']
            bin_edges = np.arange(0, np.amax(H)+new_binsize, new_binsize)
            ax.hist(H[idx_range], bins=bin_edges, label='sel.', color='tab:blue', alpha=0.5)
            ax.hist(H[~idx_range], bins=bin_edges, label='del.', color='tab:grey', alpha=0.5)
            ax.axvline(mph, c='r', lw=0.5, label='$\it{mph}=%.2f$' % (mph))
            bin_max = np.amax(H)
            ax.set_xlim([0, bin_max])
            ax.set_title('Unsmoothed heights')
        elif type == 'optimal filter':
            Hopt = self.data['Hopt']
            H = self.data['H']
            pdfx = self.data['pdfx']
            pdfy = self.data['pdfy']
            bin_max = np.amax((np.amax(H), np.amax(Hopt)))
            bin_edges = np.arange(0, bin_max+new_binsize, new_binsize)
            ax.hist(H[idx_range], bins=bin_edges, label='$\it{H}$', color='tab:blue', alpha=0.5)
            ax.hist(Hopt, bins=bin_edges, color='tab:green', alpha=0.5, label='$\it{H}_{opt}$')
            ax.plot(pdfx, pdfy, c='tab:green', label='KDE')
            ax.set_xlim([0, bin_max])
            ax.set_title('Optimal filter heights')
        elif type == 'dark smoothed':
                dark_H_smoothed = self.data['dark_H_smoothed']
                if len(dark_H_smoothed):
                    H_smoothed = self.data['H_smoothed']
                    bin_edges = np.arange(mph, np.amax(dark_H_smoothed)+new_binsize, new_binsize)
                    ax.hist(dark_H_smoothed, bins=bin_edges, label='dark', color='tab:orange', alpha=0.5)
                    ax.axvline(mph, c='r', lw=0.5, label='$\it{mph}=%.2f$' % mph)
                    ax.axvline(dark_threshold, c='r', lw=0.5, ls='-.', label='$5\/\it{\\sigma}_{dark}=%.2f$' % dark_threshold)
                    ax.set_xlim([0, np.amax(H_smoothed)])
                    ax.set_title('Dark smoothed heights')
        ax.legend()
        ax.set_xlabel('$\it{H}$')
        ax.set_ylabel('$counts$')

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
        ax.set_xlabel('$\it{f}$ $[Hz]$')
        ax.set_ylabel('$PSD\/[dBc/Hz]$')
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
    

    def load_mkid(self, file):
        loaded_dict = f.load_dictionary_from_file(file)

        if loaded_dict is not None:
            print('Dictionary succesfully loaded')
            self.data = loaded_dict
            self.settings = loaded_dict['settings']
        else:
            print('Dictionary not loaded, please input correct file')