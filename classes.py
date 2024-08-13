import functions as f
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, medfilt
import numpy as np
import pickle
import time
from copy import copy

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
            self.LT = int(LT)
            self.wl = int(wl)
            self.KID = int(kid_nr)
            self.pread = int(pread)
            self.data = str(date)
            self.name = 'LT%d_%dnm_KID%d_P%d_%s' % (LT, wl, kid_nr, pread, date)

            self.chunckwise_peakmodel = False
            self.existing_peak_model = False
            self.chuncksize = int(chuncksize)
            self.max_chuncks = None
            self.discard_saturated = bool(discard_saturated)

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
            
            self.light_files, self.light_info_files = f.get_bin_files(light_dir, kid_nr, pread)
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

            if len(self.light_info_files) != 0:
                self.f0, self.Q, self.Qc, self.Qi, self.S21_min, self.fs, self.T = f.get_info(self.light_info_files[0])
            else:                
                print('No info file obtained')
                self.f0, self.Q, self.Qc, self.Qi, self.S21_min, self.fs, self.T = 0
        tstop = time.time()
        telapsed = tstop - tstart
        print('Elapsed time: %d s' % telapsed)


    def pks_vs_sigmas(self, settings, f, stds, binsize=.5, kernel=None):
        self.settings = self.import_settings(settings)

        self.signal, self.dark_signal = self.coord_transformation()
        if kernel:
            self.signal = medfilt(self.signal, kernel)
            self.dark_signal = medfilt(self.dark_signal, kernel)
        
        binedges = np.arange(-10, 30, binsize)
        print('(1/3) Constructing noise_model')

        dark_noise_std = f.get_sigma(self.dark_signal, self.sw, self.window)
        print(dark_noise_std)
        light_noise_std = f.get_sigma(self.signal, self.sw, self.window)
        print(light_noise_std)
        self.noise_std = light_noise_std
        _, nxx, _, noises = self.noise_model(self.pw, stds[0])
        noise_std = f.get_sigma(self.signal, 0, None)
        print(noise_std)
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
        fig, ax = plt.subplot_mosaic('ca;fe;bd', figsize=(10, 7), constrained_layout=True)
        for i, nr in enumerate(stds[::-1][:-1]):
            lower_nr = stds[::-1][i+1]
            mph, mpp = self.get_mph([lower_nr, nr])

            locs, props = f.find_pks(self.signal, mph[0], mpp, self.sw, self.window, self.sff)
            neg_locs, neg_props = f.find_pks(-self.signal, mph[0], mpp, self.sw, self.window, self.sff)
            locs = np.hstack((locs, neg_locs))
            sort = np.argsort(locs)
            locs = locs[sort]
            H = props['peak_heights']
            neg_H = -neg_props['peak_heights']
            H = np.hstack((H, neg_H))[sort]
            args = np.argwhere((H >= mph[0]) & (H < mph[1])).flatten()
            neg_args = np.argwhere((H <= -mph[0]) & (H > mph[1])).flatten()
            pulses, single_idx = f.get_single_pulses(self.signal, args, locs, self.pulse_length, self.rise_offset)
            neg_pulses, neg_single_idx = f.get_single_pulses(self.signal, neg_args, locs, self.pulse_length, self.rise_offset)
            

            dark_locs, dark_props = f.find_pks(self.dark_signal, mph[0], mpp, self.sw, self.window, self.sff)
            neg_dark_locs, neg_dark_props = f.find_pks(-self.dark_signal, mph[0], mpp, self.sw, self.window, self.sff)
            dark_locs = np.hstack((dark_locs, neg_dark_locs))
            sort = np.argsort(dark_locs)
            dark_locs = dark_locs[sort]
            dark_H = dark_props['peak_heights']
            neg_dark_H = -neg_dark_props['peak_heights']
            dark_H = np.hstack((dark_H, neg_dark_H))[sort]
            dark_args = np.argwhere((dark_H >= mph[0]) & (dark_H < mph[1])).flatten()
            neg_dark_args = np.argwhere((dark_H <= mph[0]) & (dark_H < mph[1])).flatten()
            dark_pulses, dark_single_idx = f.get_single_pulses(self.dark_signal, dark_args, dark_locs, self.pulse_length, self.rise_offset)
            neg_dark_pulses, neg_dark_single_idx = f.get_single_pulses(self.dark_signal, neg_dark_args, dark_locs, self.pulse_length, self.rise_offset)
            
            # locs, props = f.find_pks(self.signal, mph, mpp[0], self.sw, self.window, self.sff)
            # neg_locs, neg_props = f.find_pks(-self.signal, mph, mpp[0], self.sw, self.window, self.sff)
            # # locs = np.hstack((locs, neg_locs))
            # # sort = np.argsort(locs)
            # # locs = locs[sort]
            # H = props['peak_heights']
            # neg_H = neg_props['peak_heights']
            # # H = np.hstack((H, neg_H))[sort]
            # args = np.argwhere((H >= mpp[0]) & (H < mpp[1])).flatten()
            # neg_args = np.argwhere((neg_H >= mpp[0]) & (neg_H < mpp[1])).flatten()
            # pulses, single_idx = f.get_single_pulses(self.signal, args, locs, self.pulse_length, self.rise_offset)
            # neg_pulses, neg_single_idx = f.get_single_pulses(self.signal, neg_args, neg_locs, self.pulse_length, self.rise_offset)
            
            # dark_locs, dark_props = f.find_pks(self.dark_signal, mph, mpp[0], self.sw, self.window, self.sff)
            # neg_dark_locs, neg_dark_props = f.find_pks(-self.dark_signal, mph, mpp[0], self.sw, self.window, self.sff)
            # # dark_locs = np.hstack((dark_locs, neg_dark_locs))
            # # sort = np.argsort(dark_locs)
            # # dark_locs = dark_locs[sort]
            # dark_H = dark_props['peak_heights']
            # neg_dark_H = neg_dark_props['peak_heights']
            # # dark_H = np.hstack((dark_H, neg_dark_H))[sort]
            # dark_args = np.argwhere((dark_H >= mpp[0]) & (dark_H < mpp[1])).flatten()
            # neg_dark_args = np.argwhere((neg_dark_H >= mpp[0]) & (neg_dark_H > mpp[1])).flatten()
            # dark_pulses, dark_single_idx = f.get_single_pulses(self.dark_signal, dark_args, dark_locs, self.pulse_length, self.rise_offset)
            # neg_dark_pulses, neg_dark_single_idx = f.get_single_pulses(self.dark_signal, neg_dark_args, neg_dark_locs, self.pulse_length, self.rise_offset)

            if np.sum(single_idx):       
                if i==0:
                    pulse_template = np.mean(pulses, axis=0) 
                    H_opt0, _, _, _ = f.optimal_filter(noises, pulse_template, self.sf, self.ssf, nxx)
                    ax['b'].hist(H_opt0/noise_std, bins=binedges, label='Noises', alpha=0.5, zorder=0, facecolor='tab:gray')
                    ax['d'].hist(H_opt0/noise_std, bins=binedges, label='Noises', alpha=0.5, zorder=0, facecolor='tab:gray')
                H_opt, _, _, _ = f.optimal_filter(pulses, pulse_template, self.sf, self.ssf, nxx)
                ax['a'].plot(np.mean(pulses, axis=0)/np.amax(pulse_template), label='%d/%d counts' % (np.sum(single_idx), len(single_idx)), zorder=i, alpha=0.5, c=colors[i])
                ax['d'].hist(H_opt/noise_std, bins=binedges, label='%d-%d$\sigma$' % (lower_nr, nr), alpha=0.5, zorder=nr, edgecolor=colors[i], facecolor='None')
            if np.sum(neg_single_idx):       
                neg_H_opt, _, _, _ = f.optimal_filter(neg_pulses, pulse_template, self.sf, self.ssf, nxx)
                ax['e'].plot(np.mean(neg_pulses, axis=0)/np.amax(pulse_template), label='%d/%d counts' % (np.sum(neg_single_idx), len(neg_single_idx)), zorder=i, alpha=0.5, c=colors[i])
                ax['d'].hist(neg_H_opt/noise_std, bins=binedges, alpha=0.5, zorder=nr, edgecolor=colors[i], facecolor='None')
            if np.sum(dark_single_idx):       
                dark_H_opt, _, _, _ = f.optimal_filter(dark_pulses, pulse_template, self.sf, self.ssf, nxx)
                ax['c'].plot(np.mean(dark_pulses, axis=0)/np.amax(pulse_template), label='%d/%d counts' % (np.sum(dark_single_idx), len(dark_single_idx)), zorder=i, alpha=0.5, c=colors[i])
                ax['b'].hist(dark_H_opt/noise_std, bins=binedges, label='%d-%d$\sigma$' % (lower_nr, nr), alpha=0.5, zorder=nr, edgecolor=colors[i], facecolor='None')
            if np.sum(neg_dark_single_idx):       
                neg_dark_H_opt, _, _, _ = f.optimal_filter(neg_dark_pulses, pulse_template, self.sf, self.ssf, nxx)
                ax['f'].plot(np.mean(neg_dark_pulses, axis=0)/np.amax(pulse_template), label='%d/%d counts' % (np.sum(neg_dark_single_idx), len(neg_dark_single_idx)), zorder=i, alpha=0.5, c=colors[i])
                ax['b'].hist(neg_dark_H_opt/noise_std, bins=binedges, label='%d-%d$\sigma$' % (lower_nr, nr), alpha=0.5, zorder=nr, edgecolor=colors[i], facecolor='None')
        ax['c'].plot(np.mean(noises, axis=0)/np.amax(pulse_template), label='noises', zorder=i+1, alpha=0.5, c='k')
        ax['a'].set_title('pulse model')
        ax['a'].set_xlabel('')
        ax['c'].set_title('Dark')
        ax['b'].set_xlabel('$H/\sigma$')
        ax['a'].set_title('Light')
        ax['d'].set_xlabel('$H/\sigma$')
        ax['a'].legend()
        ax['b'].legend()
        ax['c'].legend()
        ax['d'].legend()
        ax['e'].legend()
        ax['f'].legend()


    def overview(self, settings, f, max_chuncks=None, redo_peak_model=False, iterate=False, plot_pulses=False, save=False, figpath=''):
        print('----------------STARTED----------------')
        
        tstart = time.time()
        self.settings = self.import_settings(settings)
        self.max_chuncks=max_chuncks
        self.red_peak_model=redo_peak_model
        self.iterate=iterate
        self.plot_pulses = plot_pulses
        self.save = save
        self.figpath = figpath

        self.signal, self.dark_signal = self.coord_transformation()


        print('(1/3) Constructing noise_model')
        self.fxx, self.nxx, _, noises = self.noise_model(self.pw)
        self.Nfxx, self.Nxx, self.noise_std, _ = self.noise_model(self.max_bw)
        self.mph, self.mpp = self.get_mph(self.height, self.prominence)

        if self.existing_peak_model==False or (self.existing_peak_model==True and redo_peak_model==True): 
            print('(2/3) Constructing peak_model, aligning on pulse %s' % self.align)
            self.pulses, self.H, self.sel_locs, self.filtered_locs, self.H_smoothed = self.peak_model(self.mph, self.mpp)
            self.filter_pulses()
        else:
            print('(2/3) Reloading existing peak_model')
            print('   (%d/%d) light files processed:' % (self.nr_segments, self.nr_segments))
            self.filter_pulses()
        self.existing_peak_model = True


        # Get pulses in dark data
        print('   (%d/%d) dark files processed:' % (self.nr_dark_segments, self.nr_dark_segments))  
        _, self.dark_H, self.sel_dark_locs, self.filtered_dark_locs, self.dark_H_smoothed = self.find_peaks(self.dark_signal, self.mph, self.mpp)
        self.dark_locs = np.hstack((self.sel_dark_locs, self.filtered_dark_locs))

        ## Mean pulse
        pulses_range = self.pulses[self.idx_range, :]
        H_range = self.H[self.idx_range]
        self.mean_pulse = np.mean(pulses_range, axis=0)
        self.sxx = f.psd(self.mean_pulse, self.sf*self.ssf)[1:round(self.pw*self.sff/2)+1]


        ###########################################
        if self.iterate:
            self.H0 = copy(self.H)
            self.idx_range0 = copy(self.idx_range)
            self.H_opt0, _, _, _ = self.optimal_filter(pulses_range)

            template = self.mean_pulse
            self.window = template[::-1] / np.sum(template)
            self.sw = self.pulse_length
            self.windowtype = 'iterative'
            self.window_offset = int(np.argmax(self.window[::-1]))

            self.fxx, self.nxx, _, noises = self.noise_model(self.pw)
            self.Nfxx, self.Nxx, self.noise_std, _ = self.noise_model(self.max_bw)
            self.mph, self.mpp = self.get_mph(self.height, self.prominence)

            if self.existing_peak_model==False or (self.existing_peak_model==True and redo_peak_model==True): 
                print('(2/3) Constructing peak_model, aligning on pulse %s' % self.align)
                self.pulses, self.H, self.sel_locs, self.filtered_locs, self.H_smoothed = self.peak_model(self.mph, self.mpp)
                self.filter_pulses()
            else:
                print('(2/3) Reloading existing peak_model')
                print('   (%d/%d) light files processed:' % (self.nr_segments, self.nr_segments))
                self.filter_pulses()
            self.existing_peak_model = True

            # Get pulses in dark data
            print('   (%d/%d) dark files processed:' % (self.nr_dark_segments, self.nr_dark_segments))  
            _, self.dark_H, self.sel_dark_locs, self.filtered_dark_locs, self.dark_H_smoothed = self.find_peaks(self.dark_signal, self.mph, self.mpp)
            self.dark_locs = np.hstack((self.sel_dark_locs, self.filtered_dark_locs))

            ## Mean pulse
            pulses_range = self.pulses[self.idx_range, :]
            H_range = self.H[self.idx_range]
            self.mean_pulse = np.mean(pulses_range, axis=0)
            self.sxx = f.psd(self.mean_pulse, self.sf*self.ssf)[1:round(self.pw*self.sff/2)+1]
    
        
        ## Determine some pulse statistics
        nr_sel_pulses = len(self.sel_locs)
        if nr_sel_pulses == 0:
            raise Exception("   No pulses selected")
        nr_rej_pulses = len(self.filtered_locs)
        nr_det_pulses = nr_sel_pulses + nr_rej_pulses
        self.rej_perc = 100 * (1 - nr_sel_pulses / nr_det_pulses)
        self.photon_rate = nr_det_pulses / self.nr_segments
        self.photon_rate_range = len(H_range) / self.nr_segments
        self.dark_photon_rate = len(self.dark_locs) / self.nr_dark_segments


        ## Optimal filtering and resolving powers
        print('(3/3) Applying optimal_filter')
        self.H_opt, self.R_sn, self.mean_dxx, self.chi_sq = self.optimal_filter(pulses_range)
        self.H_0, _, _, _ = self.optimal_filter(noises)
        self.mean_H_opt = np.mean(self.H_opt)
        binedges = np.histogram_bin_edges(self.H_opt, bins='auto')
        pulse_binsize = binedges[1] - binedges[0]
        self.R, _, _, _, _ = f.resolving_power(H_range, pulse_binsize)
        self.R_opt, self.pdf_y, self.pdf_x, mu_opt, _ = f.resolving_power(self.H_opt[self.idx_range], pulse_binsize)
        self.R_i = 1 / np.sqrt(1 / self.R_opt**2 - 1 / self.R_sn**2)
        binedges = np.histogram_bin_edges(self.H_0, bins='auto')
        noise_binsize = binedges[1]-binedges[0]
        _, _, _, _, fwhm_0 = f.resolving_power(self.H_0, noise_binsize)
        self.R_0 = mu_opt / fwhm_0
        
        ## Fit lifetime
        self.tau_qp, self.fit_x, self.fit_y = self.fit_lifetime()

        ## Plot overview
        self.plot_overview()

        ## Save data
        if save:
            filename = self.save(figpath)
            print('SAVED MKID OBJECT: "%s"' % filename)

        tstop = time.time()
        telapsed = tstop - tstart
        print('----------------FINISHED (IN %d s)----------------' % telapsed)


    def import_settings(self, settings):
        self.sf = f.ensure_type(settings['sf'], int)
        self.sff = int(self.sf / 1e6)
        settings['sff'] = f.ensure_type(self.sff, int)
        self.response = f.ensure_type(settings['response'], str)
        self.coord = f.ensure_type(settings['coord'], str)
        self.pulse_length = f.ensure_type(settings['pw'], int)
        self.rise_offset = f.ensure_type(settings['rise_offset'], int)
        self.pw = int((self.pulse_length + self.rise_offset))
        self.buffer = f.ensure_type(settings['buffer'], int)
        self.windowtype = f.ensure_type(settings['window'], str)
        self.sw = f.ensure_type(settings['sw'], int)
        if self.sw and self.sw > 1:
            self.window = f.get_window(self.windowtype, self.sw*self.sff)
            self.window_offset = int(np.argmax(self.window[::-1]))
        else:
            self.sw = 0
            self.window = None
            self.window_offset = 0
        self.ssf = f.ensure_type(settings['ssf'], int)
        if self.ssf and self.ssf > 1:
            pass
        else:
            self.ssf = 1
            settings['ssf'] = 1
        self.align = f.ensure_type(settings['align'], str)
        if self.align == 'peak':
            self.ssf = 1
            settings['ssf'] = self.ssf
        self.sstype = f.ensure_type(settings['sstype'], str)
        
        self.height = f.ensure_type(settings['mph'], (int, list, tuple, np.ndarray), orNoneType=True)
        self.prominence = f.ensure_type(settings['mpp'], (int, list, tuple, np.ndarray), orNoneType=True)

        self.nr_noise_segments = f.ensure_type(settings['nr_noise_segments'], int)
        self.binsize = f.ensure_type(settings['binsize'], float)
        self.H_range = f.ensure_type(settings['H_range'], (float, list, tuple), orNoneType=True)
        self.fit_T = f.ensure_type(settings['fit_T'], (int, np.ndarray))
        self.max_bw = f.ensure_type(settings['max_bw'], int)
        self.tlim = f.ensure_type(settings['tlim'], (int, np.ndarray))
        self.filter_std = f.ensure_type(settings['filter_std'], int)
        self.noise_thres = f.ensure_type(settings['noise_thres'], (float, int))
        return settings


    def get_mph(self, min_height=None, min_prominence=None):
        if min_height == None:
            mph = [5 * self.noise_std, None]
        elif isinstance(min_height, (int, float)):
            mph = [min_height * self.noise_std, None]
        else:
            mph = [ph * self.noise_std for ph in min_height]
        if min_prominence:
            mpp = min_prominence * self.noise_std
        else:
            mpp = mph[0] / 2
        return mph, mpp


    def coord_transformation(self):
            return f.coord_transformation(self.response, self.coord, self.phase, self.amp, self.dark_phase, self.dark_amp)
    

    def peak_model(self, mph, mpp):
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
                signal = f.coord_transformation(self.response, self.coord, phase, amp)

                if self.plot_pulses and start==0:
                    pass
                else:
                    self.plot_pulses = False

                pulses_chunck, H_chunck, sel_locs_chunck, filtered_locs_chunck, H_smoothed_chunck = self.find_peaks(signal, mph, mpp)
                
                if len(sel_locs_chunck) != 0:
                    sel_locs_chunck += int(start * self.sf)
                    pulses.append(pulses_chunck)
                    H.append(H_chunck)
                    sel_locs.append(sel_locs_chunck)
                    filtered_locs.append(filtered_locs_chunck)
                    H_smoothed.append(H_smoothed_chunck)
                if len(filtered_locs_chunck) != 0:
                    filtered_locs_chunck += int(start * self.sf)              

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
            pulses, H, sel_locs, filtered_locs, H_smoothed = self.find_peaks(self.signal, mph, mpp)
        return pulses, H, sel_locs, filtered_locs, H_smoothed


    def find_peaks(self, signal, mph, mpp):
        return f.peak_model(signal, mph, mpp, self.pw, self.sw, self.align, self.window, self.sff, self.ssf, self.sstype, self.buffer, self.rise_offset, plot_pulse=self.plot_pulses)


    def noise_model(self, bw, thres=None):
        return f.noise_model(self.dark_signal, bw, self.sff, self.nr_noise_segments, self.sw, self.window, thres)
    

    def filter_pulses(self):
        if len(self.H) > 0:
            self.pulses, self.H, self.sel_locs, self.filtered_locs, self.H_smoothed, self.idx_range = f.filter_pulses(self.pulses, self.H, self.sel_locs, self.filtered_locs, self.H_smoothed, self.H_range, self.filter_std)


    def fit_lifetime(self):
        tau_qp, dtau_qp, popt = f.fit_decaytime(self.mean_pulse, self.pw, [T+self.rise_offset for T in self.fit_T])
        if isinstance(self.fit_T, (int, float)):
            plot_x = np.linspace(self.fit_T, self.pw, (self.pw - self.fit_T) * self.ssf * self.sff)
        elif isinstance(self.fit_T, (tuple, list, np.ndarray)):
            plot_x = np.linspace(self.fit_T[0], self.fit_T[1], (self.fit_T[1] - self.fit_T[0]) * self.ssf * self.sff)
        fit_x = np.arange(len(plot_x))
        plot_y = f.exp_decay(fit_x, *popt)
        return tau_qp, plot_x, plot_y


    def optimal_filter(self, pulses):
        return f.optimal_filter(pulses, self.mean_pulse, self.sf, self.ssf, self.nxx)
    

    def plot_overview(self):
        fig, axes = plt.subplot_mosaic("AABB;CDEF;GHIJ;KKKK", layout='constrained', figsize=(18, 10))
        fig.suptitle('Overview: %s' % (self.name))
        self.plot_timestream(axes['A'], 'light')
        self.plot_timestream(axes['B'], 'dark')

        if self.sw:
            self.plot_hist(axes['C'], 'dark smoothed')
            self.plot_hist(axes['D'], 'smoothed')
        self.plot_hist(axes['E'], 'unsmoothed')
        self.plot_hist(axes['F'], 'optimal filter')
        
        self.plot_stacked_pulses(axes['G'])
        self.plot_mean_pulse(axes['H'], type='lin')
        self.plot_mean_pulse(axes['I'], type='log')
        self.plot_psds(axes['J'])

        self.plot_table(axes['K'])


    def plot_timestream(self, ax, type):
        if type == 'light':
            signal = self.signal
            locs = self.sel_locs
            filtered_locs = self.filtered_locs
            ax.set_title('Light timestream')
        elif type == 'dark':   
            signal = self.dark_signal
            locs = self.dark_locs
            ax.set_title('Dark timestream')
        plot_locs_idx = (locs >= self.tlim[0]*self.sf) & (locs < self.tlim[1]*self.sf)
        plot_locs = locs[plot_locs_idx]

        t_idx = np.arange(int(self.tlim[0]*self.sf), int(self.tlim[1]*self.sf), 1)
        ylim = [-0.5, np.ceil(np.amax(signal[t_idx]))]
        t = t_idx / self.sf
        
        ax.plot(t, signal[t_idx], linewidth=0.5, label='timestream', zorder=1)  
        if self.sw:  
            len_window = len(self.window)
            smoothed_signal = np.convolve(signal, self.window, mode='valid')
            ax.plot(t[:-len_window+1], smoothed_signal[t_idx[:-len_window+1]], lw=0.5, label='smoothed', zorder=2)
        else:
            smoothed_signal = signal 
        if type == 'light':
            if len(plot_locs_idx):
                plot_sel_H = self.H_smoothed[plot_locs_idx]
                ax.scatter(t[plot_locs], plot_sel_H, marker='v', c='None', edgecolors='tab:green', lw=0.5, label='sel. pulses', zorder=4)
            plot_filtered_idx = (filtered_locs >= self.tlim[0]*self.sf) & (filtered_locs < self.tlim[1]*self.sf)
            if len(plot_filtered_idx):
                plot_filtered_locs = filtered_locs[plot_filtered_idx]
                plot_filtered_H = smoothed_signal[plot_filtered_locs]
                ax.scatter(t[plot_filtered_locs], plot_filtered_H, marker='v', c='None', edgecolors='tab:red', lw=0.5, label='del. pulses', zorder=4)
        elif type == 'dark':
            if len(plot_locs):
                plot_dark_H = smoothed_signal[plot_locs]
                ax.scatter(t[plot_locs], plot_dark_H, marker='v', c='None', edgecolors='tab:red', lw=0.5, zorder=4)
        for ph in self.mph:
            if ph:
                ax.axhline(ph, color='tab:red', lw=0.5, label='$\it{mph}=%.2f$' % (ph), zorder=3)
        ax.set_ylim(ylim)
        ax.set_xlim(self.tlim)
        ax.set_xlabel('$\it{t}$ $[s]$')
        ax.set_ylabel('$response$')
        ax.legend(bbox_to_anchor=(0., 0, 1., .102), loc='lower left',
                      ncols=5, mode="expand", borderaxespad=0., fontsize=9)


    def plot_stacked_pulses(self, ax):
        pulses = self.pulses[self.idx_range]
        nr_pulses, len_pulses = pulses.shape
        t = np.linspace(0, self.pw-1, len_pulses) - self.rise_offset
        nr2plot = 100
        if nr2plot > nr_pulses:
            every = 1
        else:
            every = int(np.round(nr_pulses / nr2plot))           
        pulses2plot = pulses[::every, :].T
        ax.plot(t, pulses2plot, lw=0.25)
        ax.set_xlim([-self.rise_offset, self.pw- self.rise_offset])
        ax.set_xlabel('$\it{t}$ $[\mu s]$')
        ax.set_ylabel('$response$')
        ax.set_title('%d overlayed pulses' % pulses2plot.shape[-1])
    

    def plot_psds(self, ax):
        ax.semilogx(self.fxx, 10*np.log10(self.sxx), label='$\it M(f)$')
        ax.semilogx(self.fxx, 10*np.log10(self.nxx), label='$\it N(f)$')
        ax.semilogx(self.fxx, 10*np.log10(self.mean_dxx), label='$\it D(f)$')
        ax.set_ylim([10*np.log10(np.amin(self.nxx)), 10*np.log10(np.amax(self.sxx))])
        ax.set_xlim([self.fxx[1], .5*self.sf])
        ax.grid(which='major', lw=0.5)
        ax.grid(which='minor', lw=0.2)
        ax.xaxis.get_major_locator().set_params(numticks=99)
        ax.xaxis.get_minor_locator().set_params(numticks=99, subs=np.arange(2, 10, 2)*.1)
        ax.legend()
        ax.set_xlabel('$\it{f}$ $[Hz]$')
        ax.set_ylabel('$PSD$ $[dBc/Hz]$')
        ax.set_title('Optimal filter models')
    

    def plot_mean_pulse(self, ax, type):
        len_mean_pulse = len(self.mean_pulse)
        t = np.linspace(-self.rise_offset, self.pulse_length-1, len_mean_pulse)
        if type == 'lin': 
            ax.plot(t, self.mean_pulse, label='mean pulse', zorder=1)
            ax.plot(self.fit_x, self.fit_y, ls='--', label='fit', zorder=2)
            ax.set_title('Average pulse')
        elif type == 'log':
            ax.semilogy(t, self.mean_pulse, label='mean pulse', zorder=1)
            ax.semilogy(self.fit_x, self.fit_y, ls='--', label='fit', zorder=2)
            ax.set_title('Average pulse semilog')
            ax.set_ylim([1e-3, np.ceil(np.amax(self.mean_pulse))])
        ax.set_xlim([-self.rise_offset, self.pulse_length])
        ax.set_xlabel('$\it{t}$ $[\mu s]$')
        ax.set_ylabel('$response$')
        ax.legend()


    def plot_hist(self, ax, type):
        bin_max = np.amax((self.H, self.H_opt))
        if type=='smoothed':
            bin_edges = np.histogram_bin_edges(self.H_smoothed, bins='auto')
            ax.hist(self.H_smoothed[self.idx_range], bins=bin_edges, label='sel.', color='tab:orange', alpha=0.5)
            ax.hist(self.H_smoothed[~self.idx_range], bins=bin_edges, label='del.', color='tab:grey', alpha=0.5) 
            if self.H_range:
                if isinstance(self.H_range, (int, float)):
                    ax.axvline(self.H_range, c='r', lw=0.5, ls='--', label='$limit$')
                elif isinstance(self.H_range, (tuple, list, np.ndarray)):
                    ax.axvline(self.H_range[0], c='r', lw=0.5, ls='--', label='$limit$')
                    ax.axvline(self.H_range[1], c='r', lw=0.5, ls='--')
            for ph in self.mph:
                if ph:
                    ax.axvline(ph, c='r', lw=0.5, label='$\it{mph}=%.2f$' % (ph))
            ax.set_title('Smoothed heights')    
        elif type=='unsmoothed':
            bin_edges = np.histogram_bin_edges(self.H, bins='auto')
            ax.hist(self.H[self.idx_range], bins=bin_edges, label='sel.', color='tab:blue', alpha=0.5)
            ax.hist(self.H[~self.idx_range], bins=bin_edges, label='del.', color='tab:grey', alpha=0.5)
            if self.iterate:
                ax.hist(self.H0[self.idx_range0], bins=bin_edges, label='sel.', color='tab:red', alpha=0.5)
                ax.hist(self.H0[~self.idx_range0], bins=bin_edges, label='del.', color='tab:grey', alpha=0.5)
            for ph in self.mph:
                if ph:
                    ax.axvline(ph, c='r', lw=0.5, label='$\it{mph}=%.2f$' % (ph))
            ax.set_title('Unsmoothed heights')
        elif type == 'optimal filter':
            bin_edges = np.histogram_bin_edges(self.H_0, bins='auto')
            ax.hist(self.H_0, color='k', alpha=0.5, label='$\it{H}_{0}$')
            bin_edges = np.histogram_bin_edges(self.H_opt, bins='auto')
            ax.hist(self.H[self.idx_range], bins=bin_edges, label='$\it{H}$', color='tab:blue', alpha=0.5)
            ax.hist(self.H_opt, bins=bin_edges, color='tab:green', alpha=0.5, label='$\it{H}_{opt}$')
            ax.plot(self.pdf_x, self.pdf_y, c='k', ls='--', lw=0.5, label='KDE')
            ax.set_title('Optimal filter heights')
        elif type == 'dark smoothed':
                if len(self.dark_H_smoothed):
                    bin_edges = np.histogram_bin_edges(self.dark_H_smoothed, bins='auto')
                    ax.hist(self.dark_H_smoothed, bins=bin_edges, label='dark', color='tab:orange', alpha=0.5)
                    for ph in self.mph:
                        if ph:
                            ax.axvline(ph, c='r', lw=0.5, label='$\it{mph}=%.2f$' % (ph))
                    ax.axvline(self.noise_std, c='r', lw=0.5, ls='-.', label='$5\/\it{\\sigma}_{dark}=%.2f$' % self.noise_std)
                    ax.set_title('Dark smoothed heights')
        ax.legend()
        ax.set_xlim([0, bin_max])
        ax.set_xlabel('$\it{H}$')
        ax.set_ylabel('$counts$')


    def plot_psd_noise(self, ax):
        ax.semilogx(self.Nfxx, 10*np.log10(self.Nxx))
        ax.set_ylim([-100, 10*np.log10(np.amax(self.Nxx))])
        ax.grid(which='major', lw=0.5)
        ax.grid(which='minor', lw=0.2)
        ax.xaxis.get_major_locator().set_params(numticks=99)
        ax.xaxis.get_minor_locator().set_params(numticks=99, subs=np.arange(2, 10, 2)*.1)
        ax.legend()
        ax.set_xlabel('$\it{f}$ $[Hz]$')
        ax.set_ylabel('$PSD\/[dBc/Hz]$')
        ax.set_title('PSD dark')
    

    def plot_table(self, ax):
        # table_results = ['T', 'nr_segments', 'Q', 'Qi', 'Qc', 'Nph_dark', 'Nph', 'rej.', '<Hopt>', 'R', 'Ropt', 'Ri', 'Rsn', 'R0', 'tqp']
        table_results = [self.T, self.nr_segments, self.Q, self.Qi, self.Qc, self.dark_photon_rate, self.photon_rate, self.rej_perc, self.mean_H_opt, self.R, self.R_opt, self.R_i, self.R_sn, self.R_0, self.tau_qp]
        col_labels = ['$\it{T}$ $[K]$', '$\it{T}$ $[s]$', '$\it{Q}$', '$\it{Q_i}$', '$\it{Q_c}$', '$\it{N}_{ph}^{dark}$ $[cps]$', '$\it{N}_{ph}^{det}$ $[cps]$', 'rej. [%]', '$<\it{H}_{opt}>$ $[rad]$', '$\it{R}$', '$\it{R}_{opt}$', '$\it{R}_i$', '$\it{R}_{sn}$', '$\it{R}_{0}$', '$\it{\\tau}_{qp}$ $[\mu s]$']
        table_formats = ['%.1f', '%d', '%.1e', '%.1e', '%.1e', '%.1f', '%.f', '%.f', '%.1f', '%.1f', '%.1f', '%.1f', '%.1f', '%.1f','%.f']
        results_text = [[table_formats[i] % table_results[i] for i in np.arange(len(table_results))]] 
        the_table = ax.table(cellText=results_text,
                    rowLabels=['results'],
                    colLabels=col_labels,
                    loc='center')
        _ = ax.axis('off')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(10)


    def save(self, figpath):
        fname = figpath + self.name
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