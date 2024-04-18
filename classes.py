import functions as f
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

try:
    plt.style.use('../../mpl-stylesheet/custom_mpl_style/matplotlibrc')
except:
    pass


class MKID:
    def __init__(self, *args, file_path=None, discard_saturated=True):
        tstart = time.time()
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
            self.chuncksize = chuncksize
            self.max_chuncks = None
            self.discard_saturated = discard_saturated

            self.dark_files, _ = f.get_bin_files(dark_dir, kid_nr, pread)
            nr_dark_loaded = len(self.dark_files)
            if nr_dark_loaded > chuncksize:
                self.dark_files = self.dark_files[:self.chuncksize]
                self.nr_dark_segments = self.chuncksize
            else:
                self.nr_dark_segments = nr_dark_loaded     
            self.dark_amp, self.dark_phase, removed_dark = f.concat_vis(self.dark_files, discard=self.discard_saturated)
            self.nr_dark_segments -= removed_dark      
            print('%d/%d dark files loaded (%d found, %d discarded)' % (self.nr_dark_segments, self.nr_dark_segments+removed_dark, nr_dark_loaded, removed_dark))    
            if self.nr_dark_segments == 0:
                raise ValueError('No dark files obtained')
            
            self.light_files, self.light_info_file = f.get_bin_files(light_dir, kid_nr, pread)
            self.nr_light_loaded = len(self.light_files)
            if self.nr_light_loaded <= self.chuncksize:
                self.amp, self.phase, removed_light = f.concat_vis(self.light_files, discard=self.discard_saturated)
                self.nr_segments = self.nr_light_loaded - removed_light
                self.chunckwise_peakmodel = False
            else:
                start = 0
                stop = self.chuncksize
                light_files_chunck = self.light_files[start:stop]
                self.amp, self.phase, removed_light = f.concat_vis(light_files_chunck, discard=self.discard_saturated)
                self.nr_segments = self.chuncksize - removed_light
                self.chunckwise_peakmodel = True
            print('%d/%d light files loaded (%d found, %d discarded)' % (self.nr_segments, self.nr_segments+removed_light, self.nr_light_loaded, removed_light))    
            if self.nr_segments == 0:
                raise ValueError('No ligth files obtained')
            
            print('Chunckwise peakmodel is %s' % self.chunckwise_peakmodel)

            if len(self.light_info_file) != 0:
                info = f.get_info(self.light_info_file[0])
            else:
                info = {'f0' : 0 , 'Q' : 0, 'Qc' : 0, 'Qi' : 0, 'S21_min' : 0, 'fs' : 0, 'T' : 0}
                print('No info file obtained')
            for x in info:
                self.data[x] = info[x]
        tstop = time.time()
        telapsed = tstop - tstart
        print('Elapsed time: %d s' % telapsed)


    def overview(self, settings, f, max_chuncks=None, redo_peak_model=False, plot_pulses=False, save=False, figpath=''):
        print('----------------STARTED----------------')
        tstart = time.time()
        sf = settings['sf']
        sff = int(sf / 1e6)
        settings['sff'] = sff
        response = settings['response']
        coord = settings['coord']
        pl = settings['pw']
        rise_offset = settings['rise_offset']
        pw = (pl + rise_offset)
        buffer = settings['buffer']
        windowtype = settings['window']
        sw = settings['sw']
        if sw and sw > 1:
            window = f.get_window(windowtype, sw*sff)
        else:
            sw = 0
            window = None
        ssf = settings['ssf']
        if ssf and ssf > 1:
            pass
        else:
            ssf = 1
            settings['ssf'] = 1
        align = settings['align'] 
        if align == 'peak':
            ssf = 1
            settings['ssf'] = ssf
        sstype = settings['sstype']
        mph = settings['mph']
        mpp = settings['mpp']
        nr_noise_segments = settings['nr_noise_segments']
        binsize = settings['binsize']
        H_range = settings['H_range']
        fit_T = np.array(settings['fit_T'])
        max_bw = settings['max_bw']
        filter_std = settings['filter_std']
        self.settings = settings


        self.signal, self.dark_signal = f.coord_transformation(response, coord, self.phase, self.amp, self.dark_phase, self.dark_amp)
        first_sec = int(1 * sf)
        self.data['signal'] = self.signal[0:first_sec]
        self.data['dark_signal'] = self.dark_signal[0:first_sec]
        print('(1/3) Constructing noise_model')
        fxx, nxx, _, noises = f.noise_model(self.dark_signal, pw, sff, ssf, sstype, nr_noise_segments, sw, window)

        Nfxx, Nxx, dark_threshold, _ = f.noise_model(self.dark_signal, max_bw, sff, 1, sstype, nr_noise_segments, sw, window)
        self.data['dark_threshold'] = dark_threshold
        if mph == None:
            mph = dark_threshold
            mpp = mph
            self.settings['mph'] = mph
            self.settings['mpp'] = mpp

        if self.existing_peak_model==False or (self.existing_peak_model==True and redo_peak_model==True) or self.max_chuncks != max_chuncks: 
            print('(2/3) Constructing peak_model, aligning on pulse %s' % align)
            self.max_chuncks = max_chuncks
            if self.max_chuncks:
                if self.max_chuncks > 1:
                    self.chunckwise_peakmodel = True
                else:
                    self.chunckwise_peakmodel = False
            elif self.max_chuncks is None:
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
                removed = 0
                if self.max_chuncks:
                    max_files = self.max_chuncks * chunck
                else: 
                    max_files = self.nr_light_loaded
                while stop <= max_files:
                    print('   (%d/%d) light files processed:' % (stop, max_files))
                    light_files_chunck = self.light_files[start:stop]
                    if start == 0:
                        amp, phase = self.amp, self.phase
                    else:
                        amp, phase, removed_chunck = f.concat_vis(light_files_chunck, discard=self.discard_saturated)
                        removed += removed_chunck
                    signal = f.coord_transformation(response, coord, phase, amp)

                    if plot_pulses and start==0:
                        pass
                    else:
                        plot_pulses = False

                    pulses_chunck, H_chunck, sel_locs_chunck, filtered_locs_chunck, H_smoothed_chunck = f.peak_model(signal, mph, mpp, pw, sw, align, window, sff, ssf, sstype, buffer, rise_offset, plot_pulse=plot_pulses)
                    
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
                    if start >= max_files:
                        stop = max_files + 1
                    elif stop > max_files:
                        stop = max_files
                    
                self.nr_segments = max_files - removed
                pulses = np.concatenate(pulses)
                H = np.concatenate(H)
                sel_locs = np.concatenate(sel_locs)
                filtered_locs = np.concatenate(filtered_locs)
                H_smoothed = np.concatenate(H_smoothed)
            else:  
                print('   (%d/%d) light files processed:' % (self.nr_segments, self.nr_segments))  
                pulses, H, sel_locs, filtered_locs, H_smoothed = f.peak_model(self.signal, mph, mpp, pw, sw, align, window, sff, ssf, sstype, buffer, rise_offset, plot_pulse=plot_pulses)  

            # Filter the pulses that satisfy H_range with filter_std number of standard deviations
            if H_range:
                pulses, H, sel_locs, filtered_locs, H_smoothed, idx_range = f.filter_pulses(pulses, H, sel_locs, filtered_locs, H_smoothed, H_range, filter_std)
            else:
                idx_range = np.ones(len(H_smoothed), dtype=bool)

            # Save the data
            self.data['pulses'] = pulses
            self.data['sel_locs'] = sel_locs
            self.data['filtered_locs'] = filtered_locs
            self.data['H'] = H
            self.data['H_smoothed'] = H_smoothed
            self.data['idx_range'] = idx_range
            self.existing_peak_model = True
        else:
            # Load the data
            print('(2/3) Reloading existing peak_model')
            print('   (%d/%d) light files processed:' % (self.nr_segments, self.nr_segments))
            pulses = self.data['pulses']
            sel_locs = self.data['sel_locs']
            filtered_locs = self.data['filtered_locs']
            H = self.data['H']
            H_smoothed = self.data['H_smoothed']

            # Filter the pulses that satisfy H_range with filter_std number of standard deviations
            pulses, H, sel_locs, filtered_locs, H_smoothed, idx_range = f.filter_pulses(pulses, H, sel_locs, filtered_locs, H_smoothed, H_range, filter_std)
            self.data['pulses'] = pulses
            self.data['sel_locs'] = sel_locs
            self.data['filtered_locs'] = filtered_locs
            self.data['H'] = H
            self.data['H_smoothed'] = H_smoothed
            self.data['idx_range'] = idx_range

        # Get pulses in dark data
        print('   (%d/%d) dark files processed:' % (self.nr_dark_segments, self.nr_dark_segments))  
        _, dark_H, sel_dark_locs, filtered_dark_locs, dark_H_smoothed = f.peak_model(self.dark_signal, mph, mpp, pw, sw, align, window, sff, ssf, sstype, buffer, rise_offset)
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
            raise Exception("   No pulses selected")
        nr_rej_pulses = len(filtered_locs)
        nr_det_pulses = nr_sel_pulses + nr_rej_pulses
        rej_perc = 100 * (1 - nr_sel_pulses / nr_det_pulses)
        photon_rate = nr_det_pulses / self.nr_segments
        photon_rate_range = len(H_range) / self.nr_segments
        dark_photon_rate = len(dark_locs) / self.nr_dark_segments

        ## Optimal filtering and resolving powers
        print('(3/3) Applying optimal_filter')
        H_opt, R_sn, mean_dxx, chi_sq = f.optimal_filter(pulses_range, mean_pulse, sf, ssf, nxx)
        H_0, _, _, _ = f.optimal_filter(noises, mean_pulse, sf, ssf, nxx)
        mean_H_opt = np.mean(H_opt)
        R, _, _, _, _ = f.resolving_power(H_range, binsize)
        R_opt, pdf_y, pdf_x, mu_opt, _ = f.resolving_power(H_opt, binsize)
        R_i = 1 / np.sqrt(1 / R_opt**2 - 1 / R_sn**2)
        _, _, _, _, fwhm_0 = f.resolving_power(H_0, binsize)
        R_0 = mu_opt / fwhm_0
        
        ## Fit lifetime
        tau_qp, dtau_qp, popt = f.fit_decaytime(mean_pulse, pw, fit_T)
        if isinstance(fit_T, (int, float)):
            plot_x = np.linspace(fit_T, pw, (pw - fit_T) * ssf * sff) - rise_offset
        elif isinstance(fit_T, (tuple, list, np.ndarray)):
            plot_x = np.linspace(fit_T[0], fit_T[1], (fit_T[1] - fit_T[0]) * ssf * sff) - rise_offset
        fit_x = np.arange(len(plot_x))
        fit_y = f.exp_decay(fit_x, *popt)

        ## Add data and settings to MKID object
        self.data['window'] = window
        self.data['mean_pulse'] = mean_pulse
        self.data['R'] = R
        self.data['Ropt'] = R_opt
        self.data['Ri'] = R_i
        self.data['Rsn'] = R_sn
        self.data['R0'] = R_0
        self.data['chi_sq'] = chi_sq
        self.data['H0'] = H_0
        self.data['dark_H'] = dark_H
        self.data['dark_H_smoothed'] = dark_H_smoothed
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
            filename = self.save(figpath)
            print('SAVED MKID OBJECT: "%s"' % filename)

        tstop = time.time()
        telapsed = tstop - tstart
        print('----------------FINISHED (IN %d s)----------------' % telapsed)

    def plot_overview(self):
        sw = self.settings['sw']
        tlim = self.settings['tlim']
        binsize = self.settings['binsize']
        fig, axes = plt.subplot_mosaic("AABB;CDEF;GHIJ;KKKK", layout='constrained', figsize=(18, 10))
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
        pl = self.settings['pw']
        rise_offset = self.settings['rise_offset']
        pw = pl + rise_offset
        mph = self.settings['mph']
        sw = self.settings['sw']
        H = self.data['H']
        H_smoothed = self.data['H_smoothed']
        window = self.data['window']

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
        plot_locs_idx = (locs >= tlim[0]*sf) & (locs < tlim[1]*sf)
        plot_locs = locs[plot_locs_idx]

        t_idx = np.arange(int(tlim[0]*sf), int(tlim[1]*sf), 1)
        ylim = [-0.5, np.ceil(np.amax(signal[t_idx]))]
        t = t_idx / sf
        
        ax.plot(t, signal[t_idx], linewidth=0.5, label='timestream', zorder=1)  
        if sw:  
            len_window = len(window)
            smoothed_signal = np.convolve(signal, window, mode='valid')
            ax.plot(t[:-len_window+1], smoothed_signal[t_idx[:-len_window+1]], lw=0.5, label='smoothed', zorder=2)
        else:
            smoothed_signal = signal
        
        if type == 'light':
            if len(plot_locs_idx):
                plot_sel_H = H_smoothed[plot_locs_idx]
                ax.scatter(t[plot_locs], plot_sel_H, marker='v', c='None', edgecolors='tab:green', lw=0.5, label='sel. pulses', zorder=4)
            plot_filtered_idx = (filtered_locs >= tlim[0]*sf) & (filtered_locs < tlim[1]*sf)
            if len(plot_filtered_idx):
                plot_filtered_locs = filtered_locs[plot_filtered_idx]
                plot_filtered_H = smoothed_signal[plot_filtered_locs]
                ax.scatter(t[plot_filtered_locs], plot_filtered_H, marker='v', c='None', edgecolors='tab:red', lw=0.5, label='del. pulses', zorder=4)
        elif type == 'dark':
            if len(plot_locs):
                plot_dark_H = smoothed_signal[plot_locs]
                ax.scatter(t[plot_locs], plot_dark_H, marker='v', c='None', edgecolors='tab:red', lw=0.5, zorder=4)
        ax.axhline(mph, color='tab:red', lw=0.5, label='$\it{mph}=%.2f$' % (mph), zorder=3)
        ax.set_ylim(ylim)
        ax.set_xlim(tlim)
        ax.set_xlabel('$\it{t}$ $[s]$')
        ax.set_ylabel('$response$')
        ax.legend(bbox_to_anchor=(0., 0, 1., .102), loc='lower left',
                      ncols=5, mode="expand", borderaxespad=0., fontsize=9)


    def plot_stacked_pulses(self, ax):
        pw = self.settings['pw']
        rise_offset = self.settings['rise_offset']
        pulses = self.data['pulses']
        idx_range = self.data['idx_range']
        pulses = pulses[idx_range]
        nr_pulses, len_pulses = pulses.shape
        t = np.linspace(-rise_offset, pw-1, len_pulses)
        nr2plot = 100
        if nr2plot > nr_pulses:
            every = 1
        else:
            every = int(np.round(nr_pulses / nr2plot))           
        pulses2plot = pulses[::every, :].T
        ax.plot(t, pulses2plot, lw=0.25)
        ax.set_xlim([-rise_offset, pw])
        ax.set_xlabel('$\it{t}$ $[\mu s]$')
        ax.set_ylabel('$response$')
        ax.set_title('%d overlayed pulses' % pulses2plot.shape[-1])
    

    def plot_psds(self, ax):
        len_pulse = len(self.data['mean_pulse'])
        sxx = self.data['sxx']
        fxx = self.data['fxx']
        nxx = self.data['nxx']
        mean_dxx = self.data['mean_dxx']
        onesided = round(len_pulse / 2) + 1
        ax.semilogx(fxx[1:onesided], 10*np.log10(sxx[1:onesided]), label='$\it M(f)$')
        ax.semilogx(fxx[1:onesided], 10*np.log10(nxx[1:onesided]), label='$\it N(f)$')
        ax.semilogx(fxx[1:onesided], 10*np.log10(mean_dxx[1:onesided]), label='$\it D(f)$')
        ax.set_ylim([10*np.log10(np.amin(nxx)), 10*np.log10(np.amax(sxx))])
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
        rise_offset = self.settings['rise_offset']
        mean_pulse = self.data['mean_pulse']
        fitx = self.data['fitx']
        fity = self.data['fity']
        len_mean_pulse = len(mean_pulse)
        t = np.linspace(-rise_offset, pw-1, len_mean_pulse)
        if type == 'lin': 
            ax.plot(t, mean_pulse, label='mean pulse', zorder=1)
            ax.plot(fitx, fity, ls='--', label='fit', zorder=2)
            ax.set_title('Average pulse')
        elif type == 'log':
            ax.semilogy(t, mean_pulse, label='mean pulse', zorder=1)
            ax.semilogy(fitx, fity, ls='--', label='fit', zorder=2)
            ax.set_title('Average pulse semilog')
        ax.set_xlim([-rise_offset, pw])
        ax.set_xlabel('$\it{t}$ $[\mu s]$')
        ax.set_ylabel('$response$')
        ax.legend()


    def plot_hist(self, ax, type, binsize):
        mph = self.settings['mph']
        dark_threshold = self.data['dark_threshold']
        idx_range = self.data['idx_range']
        H_range = self.settings['H_range']
        if H_range:
            if isinstance(H_range, (int, float)):
                lim = H_range 
            elif isinstance(H_range, (tuple, list, np.ndarray)):
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
                elif isinstance(H_range, (tuple, list, np.ndarray)):
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
            H0 = self.data['H0']
            pdfx = self.data['pdfx']
            pdfy = self.data['pdfy']
            bin_max = np.amax((np.amax(H), np.amax(Hopt)))
            bin_edges = np.arange(0, bin_max+new_binsize, new_binsize)
            ax.hist(H0, bins=bin_edges, color='k', alpha=0.5, label='$\it{H}_{0}$')
            ax.hist(H[idx_range], bins=bin_edges, label='$\it{H}$', color='tab:blue', alpha=0.5)
            ax.hist(Hopt, bins=bin_edges, color='tab:green', alpha=0.5, label='$\it{H}_{opt}$')
            ax.plot(pdfx, pdfy, c='k', ls='--', lw=0.5, label='KDE')
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
    

    def plot_table(self, ax):
        table_results = ['T', 'nr_segments', 'Q', 'Qi', 'Qc', 'Nph_dark', 'Nph', 'rej.', '<Hopt>', 'R', 'Ropt', 'Ri', 'Rsn', 'R0', 'tqp']
        col_labels = ['$\it{T}$ $[K]$', '$\it{T}$ $[s]$', '$\it{Q}$', '$\it{Q_i}$', '$\it{Q_c}$', '$\it{N}_{ph}^{dark}$ $[cps]$', '$\it{N}_{ph}^{det}$ $[cps]$', 'rej. [%]', '$<\it{H}_{opt}>$ $[rad]$', '$\it{R}$', '$\it{R}_{opt}$', '$\it{R}_i$', '$\it{R}_{sn}$', '$\it{R}_{0}$', '$\it{\\tau}_{qp}$ $[\mu s]$']
        table_formats = ['%.1f', '%d', '%.1e', '%.1e', '%.1e', '%.1f', '%.f', '%.f', '%.1f', '%.1f', '%.1f', '%.1f', '%.1f', '%.1f','%.f']
        results_text = [[table_formats[i] % self.data[table_results[i]] for i in np.arange(len(table_results))]] 
        the_table = ax.table(cellText=results_text,
                    rowLabels=['results'],
                    colLabels=col_labels,
                    loc='center')
        _ = ax.axis('off')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)


    def save(self, figpath):
        fname = figpath + self.data['name']
        plt.savefig(fname + '_overview.png')
        plt.savefig(fname + '_overview.svg')
        with open(fname + '_settings.txt','w') as f: 
            for key, value in self.settings.items():
                f.write('%s:%s\n' % (key, value))
        with open(fname + '_data.txt', 'wb') as file:
            pickle.dump(self.data, file)
        return fname + '_data.txt'
    

    def load_mkid(self, file):
        loaded_dict = f.load_dictionary_from_file(file)

        if loaded_dict is not None:
            print('Dictionary succesfully loaded')
            self.data = loaded_dict
            self.settings = loaded_dict['settings']
        else:
            print('Dictionary not loaded, please input correct file')