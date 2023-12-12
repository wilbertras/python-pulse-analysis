import functions as f
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, welch
import pickle
import matplotlib as mpl
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['axes.prop_cycle'] = mpl.rcParamsDefault['axes.prop_cycle']


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


    def initiate(self, settings, f, binsize=0.1, dpulse=10, plot_pulse=False, every=100, below=False):
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
        
        pulses, H, sel_locs, filtered_locs, H_smoothed = f.peak_model(self.signal, mph, mpp, pw, sw, window, ssf, buffer, filter_std, rise_offset, plot_pulse, every, below)    

        t_idx = np.arange(int(tlim[0]*sf), int(tlim[1]*sf), 1)
        ylim = [-0.5, np.ceil(np.amax(self.signal[t_idx]))]
        xlim = [0, pw]
        t = np.linspace(tlim[0], tlim[1], len(t_idx))
        
        fig, axes = plt.subplot_mosaic("ABC;DEF", layout='constrained')
        fig.suptitle('Overview: %s' % (self.data['name']))
        axes["A"].plot(t, self.signal[t_idx], linewidth=0.5)  
        axes["A"].hlines(mph, *tlim, color='tab:red', lw=0.5)
        axes["A"].set_ylim(ylim)
        axes["A"].set_xlim(tlim)
        axes["B"].plot(t, self.dark_signal[t_idx], linewidth=0.5)
        axes["B"].hlines(noise_mph, *tlim, color='tab:red', lw=0.5)
        axes["B"].set_ylim(ylim)
        axes["B"].set_xlim(tlim)
        
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

        ylim = [1e-4, 3]
        axes["D"].semilogy(t, np.mean(pulses, axis=0))
        axes["D"].set_ylim(ylim)
        axes["D"].set_xlim(xlim)
        binedges = np.arange(np.amin(H), np.amax(H), binsize)
        axes["E"].hist(H, bins=binedges)
        binedges = np.arange(mph, np.amax(H_smoothed), binsize)
        axes["F"].hist(H_smoothed, bins=binedges)
        

    def plot_dark(self, settings, f):
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

        Nfxx, Nxx, _, _ = f.noise_model(self.dark_phase, max_bw, sf, ssf, nr_noise_segments, noise_mph, noise_mpp, sw)
        self.data['Nfxx'] = Nfxx
        self.data['Nxx'] = Nxx
        t_idx = np.arange(int(tlim[0]*sf), int(tlim[1]*sf), 1)
        t = np.linspace(tlim[0], tlim[1], len(t_idx))
        ylim = [-0.5, 2]
        xlim = [0, pw]
        fig, axes = plt.subplot_mosaic("AB", layout='constrained')
        axes['A'].plot(t, self.dark_phase[t_idx], linewidth=0.5)
        axes['A'].hlines(noise_mph, *tlim, color='tab:red', lw=0.5)
        axes['A'].set_ylim(ylim)
        axes['A'].set_xlim(tlim)
        if sw:  
            kernel = f.get_window(window, pw, sw)
            smoothed_dark_signal = np.convolve(self.dark_phase, kernel, mode='valid')
            axes['A'].plot(t, smoothed_dark_signal[t_idx], lw=0.5)
        axes["B"].semilogx(Nfxx[1:], 10*np.log10(Nxx[1:]), color='tab:blue')
        axes["B"].grid(which='major', lw=0.3)
        axes["B"].grid(which='minor', lw=0.2)
        axes["B"].xaxis.get_major_locator().set_params(numticks=99)
        axes["B"].xaxis.get_minor_locator().set_params(numticks=99, subs=np.arange(2, 10, 2)*.1)
        axes["B"].set_ylim([-80, 10*np.log10(np.amax(Nxx))])


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
        self.data['signal'] = self.signal[0:1000000]
        self.data['dark_signal'] = self.dark_signal[0:1000000]

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

        pulses_range = pulses[idx_range, :]
        H_range = H[idx_range]
        mean_pulse = np.mean(pulses_range, axis=0)
        
        nr_sel_pulses = len(sel_locs)
        if nr_sel_pulses == 0:
            raise Exception("No pulses selected")
        nr_rej_pulses = len(filtered_locs)
        nr_det_pulses = nr_sel_pulses + nr_rej_pulses
        rej_perc = 100 * (1 - nr_sel_pulses / nr_det_pulses)
        photon_rate = nr_det_pulses / self.nr_segments
        H_opt, R_sn, mean_dxx = f.optimal_filter(pulses_range, mean_pulse, sf, ssf, nxx)
        mean_H_opt = np.mean(H_opt)
        # fxx, sxx =  welch(mean_pulse, fs=sf*ssf, window='hamming', nperseg=pw*ssf, noverlap=None, nfft=None, return_onesided=True) # f.psd(mean_pulse, sf*ssf, return_onesided=True)
        sxx = f.psd(mean_pulse, sf*ssf)
        # fxx, dxx =  welch(pulses, fs=sf*ssf, window='hamming', nperseg=pw*ssf, noverlap=None, nfft=None, return_onesided=True, axis=1) # f.psd(mean_pulse, sf*ssf, return_onesided=True)
        # mean_dxx = np.mean(dxx, axis=0)
        R, _, _ = f.resolving_power(H_range, binsize)
        R_opt, pdf_y, pdf_x = f.resolving_power(H_opt, binsize)
        R_i = 1 / np.sqrt(1 / R_opt**2 - 1 / R_sn**2)
        
        tau_qp, dtau_qp, popt = f.fit_decaytime(mean_pulse, pw, fit_T)
        if isinstance(fit_T, (int, float)):
            plot_x = np.linspace(fit_T, pw, (pw - fit_T) * ssf)
        elif isinstance(fit_T, (tuple, list)):
            plot_x = np.linspace(fit_T[0], fit_T[1], (fit_T[1] - fit_T[0]) * ssf)
        fit_x = np.arange(len(plot_x))
        fit_y = f.exp_decay(fit_x, *popt)
        
        self.data['mean_pulse'] = mean_pulse
        self.data['R'] = R
        self.data['Ropt'] = R_opt
        self.data['Ri'] = R_i
        self.data['Rsn'] = R_sn
        self.data['H'] = H
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
        self.data['Nfxx'] = Nfxx
        self.data['Nxx'] = Nxx
        self.data['tqp'] = tau_qp
        self.data['fitx'] = plot_x
        self.data['fity'] = fit_y
        self.settings = settings

        if ssf and ssf > 1:
            pass
        else:
            ssf = 1

        t_idx = np.arange(int(tlim[0]*sf), int(tlim[1]*sf), 1)
        ylim = [-0.5, np.ceil(np.amax(self.signal[t_idx]))]
        xlim = [0, pw]
        t = np.linspace(tlim[0], tlim[1], len(t_idx))

        fig, axes = plt.subplot_mosaic("ABC;DEF;GHI;JJJ", layout='constrained', figsize=(12, 8))
        fig.suptitle('Overview: %s' % (self.data['name']))
        axes["A"].plot(t, self.signal[t_idx], linewidth=0.5)  
        axes["A"].hlines(mph, *tlim, color='tab:red', lw=0.5)
        axes["A"].set_ylim(ylim)
        axes["A"].set_xlim(tlim)
        axes["B"].plot(t, self.dark_signal[t_idx], linewidth=0.5)
        axes["B"].hlines(noise_mph, *tlim, color='tab:red', lw=0.5)
        axes["B"].set_ylim(ylim)
        axes["B"].set_xlim(tlim)
        
        if sw:  
            kernel = f.get_window(window, pw, sw)
            smoothed_signal = np.convolve(self.signal, kernel, mode='valid')
            smoothed_dark_signal = np.convolve(self.dark_signal, kernel, mode='valid')
            axes["A"].plot(t, smoothed_signal[t_idx], lw=0.5)
            axes["B"].plot(t, smoothed_dark_signal[t_idx], lw=0.5)
        t = np.linspace(0, pw-1, pw*ssf)
        axes["C"].plot(t, pulses[::10, :].T, lw=0.25)
        axes["C"].set_ylim(ylim)
        axes["C"].set_xlim(xlim)

        plot_sel_idx = (sel_locs > tlim[0]*sf) & (sel_locs < tlim[1]*sf)
        plot_sel_locs = sel_locs[plot_sel_idx]
        plot_sel_H = H[plot_sel_idx]
        axes["A"].scatter(plot_sel_locs / sf, plot_sel_H, marker='v', c='None', edgecolors='tab:green', lw=0.5)
        if len(filtered_locs) != 0:
            plot_filtered_idx = (filtered_locs > tlim[0]*sf) & (filtered_locs < tlim[1]*sf)
            plot_filtered_locs = filtered_locs[plot_filtered_idx]
            axes["A"].scatter(plot_filtered_locs / sf, self.signal[plot_filtered_locs], marker='v', c='None', edgecolors='tab:red', lw=0.5)
    
        if len(dark_locs) != 0:
            plot_dark_locs = dark_locs[(dark_locs > tlim[0]*sf) & (dark_locs < tlim[1]*sf)]
            axes["B"].scatter(plot_dark_locs / sf, self.dark_signal[plot_dark_locs], marker='v', c='None', edgecolors='tab:red', lw=0.5)
        ylim = [-100, 10*np.log10(np.amax((sxx, Nxx[:len(sxx)])))]
        onesided = round(pw / 2) + 1
        axes["D"].semilogx(fxx[:onesided], 10*np.log10(sxx[:onesided]))
        axes["D"].semilogx(fxx[:onesided], 10*np.log10(nxx[:onesided]))
        axes["D"].semilogx(fxx[:onesided], 10*np.log10(mean_dxx[:onesided]))
        axes["D"].set_ylim([-100, 10*np.log10(np.amax(sxx))])
        axes["D"].grid(which='major', lw=0.3)
        axes["D"].grid(which='minor', lw=0.2)
        axes["D"].xaxis.get_major_locator().set_params(numticks=99)
        axes["D"].xaxis.get_minor_locator().set_params(numticks=99, subs=np.arange(2, 10, 2)*.1)
        # axes["E"].semilogx(Nfxx[1:], 10*np.log10(Nxx[1:]), color='tab:blue')
        # axes["E"].semilogx(fxx[1:onesided], 10*np.log10(nxx[1:onesided]), color='tab:orange')
        # axes["E"].grid(which='major', lw=0.3)
        # axes["E"].grid(which='minor', lw=0.2)
        # axes["E"].xaxis.get_major_locator().set_params(numticks=99)
        # axes["E"].xaxis.get_minor_locator().set_params(numticks=99, subs=np.arange(2, 10, 2)*.1)
        # axes["E"].set_ylim([-100, 10*np.log10(np.amax(Nxx))])

        bin_min = np.amin((np.amin(H), np.amin(H_opt)))
        bin_max = np.amax((np.amax(H), np.amax(H_opt)))
        bin_edges = np.arange(bin_min, bin_max, binsize)
        smoothed_bin_edges = np.arange(bin_min, bin_max, binsize / 2)
        _, dark_H, _, _, dark_H_smoothed = f.peak_model(self.dark_signal, noise_mph, noise_mpp, pw, sw, window, ssf, buffer, filter_std, rise_offset)  
        axes["E"].hist(dark_H, label='dark', color='tab:green', alpha=0.5)
        axes["E"].hist(H[idx_range], bins=bin_edges, label='light', color='tab:blue', alpha=0.5)
        if sw:
            axes["E"].hist(dark_H_smoothed, label='dark smoothed', color='tab:purple', alpha=0.5)
        axes["E"].axvline(noise_mph, 0, 1, c='r', lw=0.5)
        axes["E"].set_xlim([0, bin_max])

        axes["F"].hist(H[idx_range], bins=bin_edges, label='light chosen', color='tab:blue', alpha=0.5)
        axes["F"].hist(H[~idx_range], bins=bin_edges, label='light leftout ', color='tab:grey', alpha=0.5)
        if sw:
            axes["F"].hist(H_smoothed, bins=smoothed_bin_edges, label='light smoothed', color='tab:purple', alpha=0.5)
        axes["F"].hist(H_opt, bins=bin_edges, label='optimal filter', color='tab:orange', alpha=0.5)
        axes["F"].axvline(mph, 0, 1, c='r', lw=0.5)
        axes["F"].set_xlim([0, bin_max])
        
        axes["G"].hist(H_opt, bins=bin_edges, color='tab:orange', alpha=0.5)
        axes["G"].plot(pdf_x, pdf_y, c='tab:orange')
        axes["G"].set_xlim([0, bin_max])
        

        xlim = [0, pw]
        ylim = [1e-4, 3]
        axes["H"].plot(t, mean_pulse)
        axes["H"].plot(plot_x, fit_y, ls='--')
        axes["H"].set_xlim([0, pw])
        axes["I"].semilogy(t, mean_pulse)
        axes["I"].semilogy(plot_x, f.exp_decay(fit_x, *popt), ls='--')
        axes["I"].set_xlim(xlim)
        axes["I"].set_ylim(ylim)
        
        table_results = ['T', 'Q', 'Qi', 'Qc', 'Nph_dark', 'Nph', 'rej.', '<Hopt>', 'R', 'Ropt', 'Ri', 'Rsn', 'tqp']
        table_formats = ['%.1f', '%.1e', '%.1e', '%.1e', '%.1f', '%.f', '%.f', '%.1f', '%.1f', '%.1f', '%.1f', '%.1f', '%.f']
        results_text = [[table_formats[i] % self.data[table_results[i]] for i in np.arange(len(table_results))]]
        
        the_table = axes["J"].table(cellText=results_text,
                    rowLabels=['results'],
                    colLabels=table_results,
                    loc='center')
        _ = axes["J"].axis('off')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(6)

        if save:
            fname = figpath + self.data['name']
            plt.savefig(fname + '_overview.png')
            plt.savefig(fname + '_overview.svg')
            with open(fname + '_settings.txt','w') as f: 
                for key, value in settings.items():
                    f.write('%s:%s\n' % (key, value))
            with open(fname + '_data.txt', 'wb') as file:
                pickle.dump(self.data, file)

            